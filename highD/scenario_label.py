#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python highD/scenario_label.py --h5 highD/baseline/test.h5 --raw_dir highD/raw --out_csv highD/baseline/scenario_labels.csv
scenario_label.py  —  mmTransformer용 scenario_labels.csv 생성 스크립트

h5 파일의 NAME 필드({rec_id}_{trackId}_{obs_frame})를 읽고,
raw highD CSV에서 lane-change / traffic-state 정보를 추출해 라벨링한다.

obs_frame = 히스토리의 마지막 프레임 (native fps 기준)
window    = [obs_frame - (T_H-1)*ds_stride, obs_frame + T_F*ds_stride]

Event labels (3-class):
    cut_in          : window 내 LC 발생 + 목표 차선 측 rear/alongside 차량 있음
    lane_change     : window 내 LC 발생 + 목표 차선 측 rear/alongside 차량 없음
    lane_following  : window 내 LC 없음

State labels (2-class):
    dense     : obs_frame 시점 STATE_SLOTS 점유율 <= STATE_THRESHOLD
    free_flow : 점유율 > STATE_THRESHOLD

Usage:
    # test split만
    python highD/scenario_label.py \\
        --h5      highD/baseline/test.h5 \\
        --raw_dir highD/raw \\
        --out_csv highD/baseline/scenario_labels.csv

    # test + val 합쳐서 하나의 CSV
    python highD/scenario_label.py \\
        --h5      highD/baseline/test.h5 highD/baseline/val.h5 \\
        --raw_dir highD/raw \\
        --out_csv highD/baseline/scenario_labels.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# preprocess.py와 동기화
TARGET_FPS = 3.0
T_H = 6    # history frames @ TARGET_FPS
T_F = 15   # future  frames @ TARGET_FPS

# 8 neighbor slots (highD tracks CSV 컬럼 순서, preprocess.py NEIGHBOR_COLS_8와 동일)
NEIGHBOR_COLS_8 = [
    "precedingId",       # slot 0
    "followingId",       # slot 1
    "leftPrecedingId",   # slot 2
    "leftAlongsideId",   # slot 3
    "leftFollowingId",   # slot 4
    "rightPrecedingId",  # slot 5
    "rightAlongsideId",  # slot 6
    "rightFollowingId",  # slot 7
]

# State label: preceding + left/right lead & alongside
STATE_SLOTS     = [0, 2, 3, 5, 6]
STATE_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# IO helpers
# ─────────────────────────────────────────────────────────────────────────────

def smart_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=";", low_memory=False)


def normalize_tracks(tracks: pd.DataFrame, xx: str) -> pd.DataFrame:
    df = tracks.copy()
    df.columns = [c.strip() for c in df.columns]
    df["recordingId"] = int(xx)
    if "trackId" not in df.columns and "id" in df.columns:
        df["trackId"] = df["id"]
    rename = {
        "leftPrecedingId":   "leftPrecedingId",
        "leftFollowingId":   "leftFollowingId",
        "rightPrecedingId":  "rightPrecedingId",
        "rightFollowingId":  "rightFollowingId",
        "rightAlsongsideId": "rightAlongsideId",   # highD 오타 수정
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    # 없는 컬럼은 0으로 채움
    for col in NEIGHBOR_COLS_8 + ["leftAlongsideId", "rightAlongsideId"]:
        if col not in df.columns:
            df[col] = 0
    if "latVelocity" not in df.columns:
        df["latVelocity"] = df.get("yVelocity", np.nan)
    for col in ["frame", "laneId", "trackId", "recordingId"]:
        if col not in df.columns:
            raise KeyError(f"normalize_tracks: missing required column '{col}'")
    df["frame"]   = pd.to_numeric(df["frame"],   errors="coerce").astype("Int64")
    df["laneId"]  = pd.to_numeric(df["laneId"],  errors="coerce")
    df["trackId"] = pd.to_numeric(df["trackId"], errors="coerce").astype("Int64")
    return df


def normalize_recmeta(recmeta: pd.DataFrame, xx: str) -> pd.DataFrame:
    rm = recmeta.copy()
    rm.columns = [c.strip() for c in rm.columns]
    if "recordingId" not in rm.columns:
        rm["recordingId"] = rm["id"] if "id" in rm.columns else int(xx)
    rm["recordingId"] = rm["recordingId"].astype(int)
    if "frameRate" not in rm.columns:
        raise KeyError("recordingMeta missing 'frameRate'")
    return rm[["recordingId", "frameRate"]].drop_duplicates()


# ─────────────────────────────────────────────────────────────────────────────
# Lane lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_lane_lookup(df: pd.DataFrame) -> Dict[int, Dict[int, int]]:
    tmp = df[["trackId", "frame", "laneId"]].copy()
    lookup: Dict[int, Dict[int, int]] = {}
    for tid, g in tmp.groupby("trackId", sort=False):
        d: Dict[int, int] = {}
        for f, l in zip(g["frame"].to_numpy(), g["laneId"].to_numpy()):
            if f is not None and not (isinstance(l, float) and np.isnan(l)):
                d[int(f)] = int(l)
        lookup[int(tid)] = d
    return lookup


def get_lane_at(lookup, track_id: int, frame: int) -> Optional[int]:
    d = lookup.get(int(track_id))
    if d is None:
        return None
    return d.get(int(frame))


def is_adjacent_lane(ego_lane: int, nb_lane: int) -> bool:
    return abs(int(ego_lane) - int(nb_lane)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Lane change detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_lane_change(w: pd.DataFrame) -> Tuple[bool, int, Optional[int]]:
    if "laneId" not in w.columns or "frame" not in w.columns:
        return False, 0, None
    df = w[["frame", "laneId"]].copy()
    df["frame"]  = pd.to_numeric(df["frame"],  errors="coerce").astype("Int64")
    df["laneId"] = pd.to_numeric(df["laneId"], errors="coerce")
    df = df.dropna().sort_values("frame")
    if len(df) < 2:
        return False, 0, None
    lane   = df["laneId"].to_numpy()
    frames = df["frame"].to_numpy()
    changed = lane[1:] != lane[:-1]
    count   = int(changed.sum())
    if count == 0:
        return False, 0, None
    first_frame = int(frames[1:][changed][0])
    return True, count, first_frame


def infer_lc_direction(w: pd.DataFrame, lc_frame: int, K: int = 5) -> Optional[str]:
    vcol = "yVelocity" if "yVelocity" in w.columns else "latVelocity"
    if vcol not in w.columns:
        return None
    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    win = df[(df["frame"] >= lc_frame - K) & (df["frame"] <= lc_frame + K)]
    v = pd.to_numeric(win[vcol], errors="coerce").dropna()
    if len(v) == 0:
        return None
    mv = float(v.mean())
    if mv > 0:
        return "right"
    if mv < 0:
        return "left"
    return None


def _to_int_id(x) -> Optional[int]:
    try:
        v = int(float(x))
        return None if v in (-1, 0) else v
    except Exception:
        return None


def check_adjacent_rear_or_alongside(
    w: pd.DataFrame,
    lc_frame: int,
    direction: str,
    lookup: Dict[int, Dict[int, int]],
    W: int = 25,
) -> bool:
    if direction == "left":
        rear_col      = "leftFollowingId"
        alongside_col = "leftAlongsideId"
    else:
        rear_col      = "rightFollowingId"
        alongside_col = "rightAlongsideId"
    df = w.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype("Int64")
    pre = df[(df["frame"] >= lc_frame - W) & (df["frame"] < lc_frame)]
    if len(pre) == 0:
        return False
    for _, row in pre.iterrows():
        f        = int(row["frame"])
        ego_lane = row.get("laneId", np.nan)
        if pd.isna(ego_lane):
            continue
        ego_lane = int(ego_lane)
        for col in (rear_col, alongside_col):
            if col not in pre.columns:
                continue
            nb_id = _to_int_id(row.get(col, 0))
            if nb_id is None:
                continue
            nb_lane = get_lane_at(lookup, nb_id, f)
            if nb_lane is None:
                continue
            if is_adjacent_lane(ego_lane, nb_lane):
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Event label (window 단위)
# ─────────────────────────────────────────────────────────────────────────────

def label_event(
    w: pd.DataFrame,
    lookup: Dict[int, Dict[int, int]],
    W_adj: int = 25,
    lc_direction_K: int = 5,
) -> Dict:
    out: Dict = {}
    has_lc, lc_count, lc_frame = detect_lane_change(w)
    out["lc_count"] = lc_count
    out["lc_frame"] = int(lc_frame) if lc_frame is not None else -1

    if not has_lc or lc_frame is None:
        out["event_label"]               = "lane_following"
        out["lc_direction"]              = "none"
        out["has_adj_rear_or_alongside"] = False
        return out

    direction = infer_lc_direction(w, lc_frame=lc_frame, K=lc_direction_K)
    out["lc_direction"] = direction if direction is not None else "unknown"

    if direction is None:
        out["event_label"]               = "lane_change"
        out["has_adj_rear_or_alongside"] = False
        return out

    has_adj = check_adjacent_rear_or_alongside(
        w=w, lc_frame=lc_frame, direction=direction, lookup=lookup, W=W_adj,
    )
    out["has_adj_rear_or_alongside"] = bool(has_adj)
    out["event_label"] = "cut_in" if has_adj else "lane_change"
    return out


# ─────────────────────────────────────────────────────────────────────────────
# State label (obs_frame 시점 neighbor 점유율)
# ─────────────────────────────────────────────────────────────────────────────

def compute_state_label(
    row_at_obs: Optional[pd.Series],
) -> Tuple[str, float]:
    """
    obs_frame 시점의 tracks 행에서 STATE_SLOTS 점유율 계산.
    row_at_obs is None이면 unknown 반환.
    """
    if row_at_obs is None:
        return "unknown", float("nan")

    state_cols = [NEIGHBOR_COLS_8[s] for s in STATE_SLOTS]
    present = sum(
        1 for col in state_cols
        if col in row_at_obs.index and _to_int_id(row_at_obs.get(col, 0)) is not None
    )
    occ = present / len(STATE_SLOTS)
    label = "dense" if occ <= STATE_THRESHOLD else "free_flow"
    return label, occ


# ─────────────────────────────────────────────────────────────────────────────
# Per-recording processing
# ─────────────────────────────────────────────────────────────────────────────

def label_recording(
    xx: str,
    raw_dir: Path,
    keys: List[Tuple[int, int]],   # [(trackId, obs_frame), ...]
    target_fps: float,
    W_adj: int,
) -> List[Dict]:
    """
    obs_frame = hist의 마지막 native 프레임.
    window    = [obs_frame - (T_H-1)*ds_stride, obs_frame + T_F*ds_stride]
    """
    tracks_path  = raw_dir / f"{xx}_tracks.csv"
    recmeta_path = raw_dir / f"{xx}_recordingMeta.csv"

    if not tracks_path.exists() or not recmeta_path.exists():
        print(f"  [SKIP] {xx}: raw files not found")
        return []

    tracks  = smart_read_csv(tracks_path)
    recmeta = smart_read_csv(recmeta_path)
    tracks_n  = normalize_tracks(tracks, xx)
    recmeta_n = normalize_recmeta(recmeta, xx)

    fr = float(recmeta_n["frameRate"].iloc[0])
    if not np.isfinite(fr) or fr <= 0:
        print(f"  [SKIP] {xx}: invalid frameRate={fr}")
        return []

    ds_stride = max(1, int(round(fr / target_fps)))

    tracks_n = tracks_n.dropna(subset=["frame"]).sort_values(["trackId", "frame"])
    lookup   = build_lane_lookup(tracks_n)
    by_tid   = {int(tid): g for tid, g in tracks_n.groupby("trackId", sort=False)}
    rid      = int(xx)
    rows: List[Dict] = []

    for tid, obs_frame in sorted(set(keys)):
        base = {
            "recordingId": rid,
            "trackId":     int(tid),
            "t0_frame":    int(obs_frame),   # evaluate.py 키와 동일 (obs_frame = META_FRAME)
        }
        g = by_tid.get(int(tid))
        if g is None:
            rows.append({**base,
                         "event_label": "unknown", "state_label": "unknown",
                         "occupancy": float("nan"), "lc_frame": -1,
                         "lc_count": 0, "lc_direction": "unknown",
                         "has_adj_rear_or_alongside": False})
            continue

        # window: obs_frame 기준 native frame 범위
        win_start = int(obs_frame) - (T_H - 1) * ds_stride
        win_end   = int(obs_frame) + T_F * ds_stride

        # downsampled frame 집합으로 필터 (native grid 위 점만)
        w_full = g[(g["frame"] >= win_start) & (g["frame"] <= win_end)].copy()
        w_full["frame"] = w_full["frame"].astype(int)

        # event: window 전체 lane-change 검사
        if len(w_full) == 0:
            ev = {"event_label": "unknown", "lc_frame": -1, "lc_count": 0,
                  "lc_direction": "unknown", "has_adj_rear_or_alongside": False}
        else:
            ev = label_event(w_full, lookup=lookup, W_adj=W_adj)

        # state: obs_frame 시점 neighbor 점유율
        obs_rows = g[g["frame"] == int(obs_frame)]
        row_at_obs = obs_rows.iloc[0] if len(obs_rows) > 0 else None
        state_label, occ = compute_state_label(row_at_obs)

        rows.append({**base, **ev, "state_label": state_label, "occupancy": occ})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_recording(args_tuple: Tuple) -> List[Dict]:
    xx, keys, raw_dir, target_fps, W_adj = args_tuple
    return label_recording(
        xx=xx, raw_dir=raw_dir, keys=keys,
        target_fps=target_fps, W_adj=W_adj,
    )


# ─────────────────────────────────────────────────────────────────────────────
# H5 reading — NAME 필드 파싱
# ─────────────────────────────────────────────────────────────────────────────

def read_h5_names(h5_paths: List[Path]) -> List[Tuple[str, int, int]]:
    """
    NAME 필드를 파싱해 [(xx, trackId, obs_frame), ...] 반환.
    NAME 형식: "{rec_id}_{trackId}_{obs_frame}"
    """
    result = []
    for p in h5_paths:
        if not p.exists():
            print(f"[WARN] h5 not found: {p}")
            continue
        with h5py.File(p, "r") as f:
            if "NAME" not in f:
                print(f"[WARN] {p}: NAME dataset not found — skip")
                continue
            names = [
                n.decode("utf-8") if isinstance(n, bytes) else str(n)
                for n in f["NAME"][:]
            ]
        for name in names:
            parts = name.split("_")
            if len(parts) < 3:
                print(f"[WARN] unexpected NAME format: {name}")
                continue
            xx       = f"{int(parts[0]):02d}"
            track_id = int(parts[1])
            obs_frame = int(parts[2])
            result.append((xx, track_id, obs_frame))
        print(f"  Loaded {p.name}: {len(names):,} samples")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="mmTransformer highD scenario label generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--h5", nargs="+", required=True,
        help="h5 파일 경로 (test.h5, val.h5 등 여러 개 가능)",
    )
    ap.add_argument(
        "--raw_dir", default="highD/raw",
        help="highD raw CSV 디렉토리 (XX_tracks.csv, XX_recordingMeta.csv 포함)",
    )
    ap.add_argument(
        "--out_csv", default=None,
        help="출력 CSV 경로. 미지정시 첫 번째 h5와 같은 디렉토리에 scenario_labels.csv 저장",
    )
    ap.add_argument("--target_fps",   type=float, default=TARGET_FPS,
                    help=f"다운샘플링 fps (preprocess.py와 동일하게 유지, default={TARGET_FPS})")
    ap.add_argument("--W_adj",        type=int,   default=25,
                    help="LC 직전 look-back 윈도우 (native frames). 25 ≈ 1sec @ 25fps")
    ap.add_argument("--num_workers",  type=int,   default=0,
                    help="병렬 프로세스 수 (0 = os.cpu_count())")
    return ap.parse_args()


def main() -> None:
    args     = parse_args()
    raw_dir  = Path(args.raw_dir)
    h5_paths = [Path(p) for p in args.h5]

    if args.out_csv is None:
        out_csv = h5_paths[0].parent / "scenario_labels.csv"
    else:
        out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── NAME 파싱 ─────────────────────────────────────────────────────────────
    print("[Step 1] h5 NAME 파싱 중...")
    entries = read_h5_names(h5_paths)
    n_total = len(entries)
    print(f"  총 {n_total:,} 샘플")

    # ── recording별로 키 묶기 ─────────────────────────────────────────────────
    keys_by_xx: Dict[str, List[Tuple[int, int]]] = {}
    for xx, tid, obs_frame in entries:
        keys_by_xx.setdefault(xx, []).append((tid, obs_frame))

    print(f"\n[Step 2] Event & State label 계산 중 ({len(keys_by_xx)} recordings)...")

    n_workers = args.num_workers if args.num_workers > 0 else os.cpu_count()
    work_items = [
        (xx, keys, raw_dir, args.target_fps, args.W_adj)
        for xx, keys in sorted(keys_by_xx.items())
    ]

    all_rows: List[Dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as exe:
        futs = {exe.submit(_process_one_recording, item): item[0]
                for item in work_items}
        for fut in tqdm(concurrent.futures.as_completed(futs),
                        total=len(futs), desc="Labeling"):
            rows = fut.result()
            all_rows.extend(rows)

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values(["recordingId", "trackId", "t0_frame"])

    col_order = [
        "recordingId", "trackId", "t0_frame",
        "event_label", "state_label", "occupancy",
        "lc_frame", "lc_count", "lc_direction", "has_adj_rear_or_alongside",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.to_csv(out_csv, index=False)
    print(f"\n[DONE] {len(df):,} labels → {out_csv}")

    if len(df) != n_total:
        print(f"[WARN] h5 samples={n_total}, labeled={len(df)} "
              f"(delta={n_total - len(df)})")

    if "event_label" in df.columns:
        print("\nEvent label counts:")
        print(df["event_label"].value_counts().to_string())

    if "state_label" in df.columns:
        print("\nState label counts:")
        print(df["state_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
