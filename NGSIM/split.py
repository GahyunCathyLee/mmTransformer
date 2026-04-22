#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split.py — Train / Val / Test split generator for NGSIM preprocessed data.

Scenario labels (from scenario_label.py)
─────────────────────────────────────────
  Event  (3-class): cut_in | lane_change | lane_following

Stratification modes  (--stratify)
────────────────────────────────────
  event        Track-level stratified split by event label.
  track        Track-level random split (no label stratification).
  none         Simple random split on window indices (no track grouping).

Split strategy
──────────────
  Scenario-based (event):
    Windows are grouped by track. A representative label is assigned to each
    track, then tracks are stratified-split. This prevents windows from the
    same track leaking across splits.

  Track-based random (track):
    Windows are grouped by track, then tracks are randomly split without any
    label stratification.

  Random (none):
    Window indices are shuffled and split directly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage examples
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # Default: event stratification, 7:1:2
  python split.py

  # Event-only stratification, custom ratio
  python split.py --stratify event --train 0.8 --val 0.1 --test 0.1

  # Track-level random split (scenario label 미사용)
  python split.py --stratify track

  # No scenario split (random, track 경계도 무시)
  python split.py --stratify none

  # Train만 event-stratify, val/test는 random
  python split.py --stratify event --no_stratify_eval

  # Custom paths
  python split.py \\
      --label_file data/NGSIM/c0/scenario_labels.csv \\
      --meta_dir   data/NGSIM/c0 \\
      --output_dir data/NGSIM/splits
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# Defaults  (NGSIM 경로 기준)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_LABEL_FILE = "data/NGSIM/c0/scenario_labels.csv"
DEFAULT_META_DIR   = "data/NGSIM/c0"
DEFAULT_OUTPUT_DIR = "data/NGSIM/splits"

DEFAULT_TRAIN = 0.7
DEFAULT_VAL   = 0.1
DEFAULT_TEST  = 0.2
SEED          = 42

# ──────────────────────────────────────────────────────────────────────────────
# Label constants
# ──────────────────────────────────────────────────────────────────────────────
EVENT_PRIORITY = ["cut_in", "lane_change", "lane_following"]


# ──────────────────────────────────────────────────────────────────────────────
# Track representative label helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rep_event(group: pd.DataFrame) -> str:
    """Track 내 windows의 event label 중 우선순위가 높은 값을 대표로."""
    unique = set(group["event_label"].dropna().unique())
    for label in EVENT_PRIORITY:
        if label in unique:
            return label
    return "lane_following"


def _build_track_df(df: pd.DataFrame, stratify_mode: str) -> pd.DataFrame:
    """
    Track 단위로 대표 레이블을 계산해 track_df를 반환합니다.

    Returns
    -------
    track_df : columns = [label, indices]
        label   — stratify key (str)
        indices — list of global_index values belonging to this track
    """
    records = []
    for (rid, tid), grp in df.groupby(["recordingId", "trackId"], sort=False):
        idx_list = grp["global_index"].tolist()

        records.append({"label": _rep_event(grp), "indices": idx_list})

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Split strategies
# ──────────────────────────────────────────────────────────────────────────────

def scenario_split(
    df: pd.DataFrame,
    stratify_mode: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    stratify_eval: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track 단위 Stratified Split.
    같은 track의 window가 split 경계를 넘지 않도록 보장합니다.

    stratify_eval=False 이면 train만 stratified split하고,
    val/test는 나머지 track을 단순 random 분할합니다.
    """
    eval_tag = "" if stratify_eval else "  |  eval=random"
    print(f"\n[Scenario-based split  |  stratify={stratify_mode}{eval_tag}]")

    if stratify_mode != "event":
        print(f"  [WARN] Unsupported scenario stratify='{stratify_mode}' — using event.")
        stratify_mode = "event"

    if "event_label" not in df.columns:
        raise ValueError("'event_label' column not found in label file.")

    print("  Building track-level representative labels …")
    track_df = _build_track_df(df, stratify_mode)

    print(f"  Unique tracks : {len(track_df):,}")
    print("  Label distribution (tracks):")
    for lbl, cnt in track_df["label"].value_counts().items():
        print(f"    {lbl:<40s} {cnt:>6,}")

    def _can_stratify(data: pd.DataFrame, test_size: float) -> bool:
        counts = data["label"].value_counts()
        if counts.empty or counts.min() < 2:
            return False
        n_classes = len(counts)
        n_test = int(np.ceil(len(data) * test_size))
        n_train = len(data) - n_test
        return n_train >= n_classes and n_test >= n_classes

    def _split(data, test_size, name):
        use_stratify = _can_stratify(data, test_size)
        if not use_stratify:
            print(f"  [WARN] {name}: too few tracks per event label — random split.")
        stratify = data["label"] if use_stratify else None
        return train_test_split(
            data, test_size=test_size, stratify=stratify, random_state=seed
        )

    train_tracks, temp_tracks = _split(track_df, test_size=1.0 - train_ratio, name="train/temp")
    val_frac = val_ratio / (1.0 - train_ratio)
    if stratify_eval:
        val_tracks, test_tracks = _split(temp_tracks, test_size=1.0 - val_frac, name="val/test")
    else:
        val_tracks, test_tracks = train_test_split(
            temp_tracks, test_size=1.0 - val_frac, stratify=None, random_state=seed
        )

    def _flatten(subset: pd.DataFrame) -> np.ndarray:
        return np.array(sorted(idx for idxs in subset["indices"] for idx in idxs))

    return _flatten(train_tracks), _flatten(val_tracks), _flatten(test_tracks)


def track_random_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Track 단위로 묶되, label stratification 없이 무작위 분할합니다.
    """
    print("\n[Track-based random split  |  stratify=track]")

    records = []
    for (rid, tid), grp in df.groupby(["recordingId", "trackId"], sort=False):
        records.append({"indices": grp["global_index"].tolist()})
    track_df = pd.DataFrame(records)

    print(f"  Unique tracks : {len(track_df):,}")

    shuffled = track_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n        = len(shuffled)
    n_train  = int(n * train_ratio)
    n_val    = int(n * val_ratio)

    train_tracks = shuffled.iloc[:n_train]
    val_tracks   = shuffled.iloc[n_train : n_train + n_val]
    test_tracks  = shuffled.iloc[n_train + n_val :]

    def _flatten(subset: pd.DataFrame) -> np.ndarray:
        return np.array(sorted(idx for idxs in subset["indices"] for idx in idxs))

    return _flatten(train_tracks), _flatten(val_tracks), _flatten(test_tracks)


def random_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Window index를 직접 무작위 분할합니다 (track 경계 무시)."""
    print("\n[Random split  |  stratify=none]")

    indices = df["global_index"].values
    rng     = np.random.default_rng(seed)
    indices = indices[rng.permutation(len(indices))]

    n       = len(indices)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return (
        np.sort(indices[:n_train]),
        np.sort(indices[n_train : n_train + n_val]),
        np.sort(indices[n_train + n_val :]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Post-split distribution summary
# ──────────────────────────────────────────────────────────────────────────────

def print_label_dist(df: pd.DataFrame, split_name: str, indices: np.ndarray) -> None:
    """Split 내 event / state 분포를 출력합니다."""
    subset = df[df["global_index"].isin(indices)]
    print(f"\n  {split_name}  ({len(indices):,} windows)")
    for col in ("event_label", "state_label"):
        if col not in subset.columns:
            continue
        vc  = subset[col].value_counts()
        row = "  ".join(f"{k}:{v}" for k, v in vc.items())
        print(f"    {col:<15s}: {row}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate train/val/test index splits for NGSIM mmap dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ─────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--label_file", default=DEFAULT_LABEL_FILE,
        help="Path to scenario_labels.csv produced by scenario_label.py "
             "(not required when --stratify track or none)",
    )
    ap.add_argument(
        "--meta_dir", default=DEFAULT_META_DIR,
        help="Directory containing meta_recordingId.npy and meta_trackId.npy "
             "(used when --stratify track or none)",
    )
    ap.add_argument(
        "--output_dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory to save train/val/test index .npy files",
    )

    # ── split ratios ──────────────────────────────────────────────────────────
    ap.add_argument("--train", type=float, default=DEFAULT_TRAIN, dest="train_ratio",
                    help="Training set proportion (unnormalized)")
    ap.add_argument("--val",   type=float, default=DEFAULT_VAL,   dest="val_ratio",
                    help="Validation set proportion (unnormalized)")
    ap.add_argument("--test",  type=float, default=DEFAULT_TEST,  dest="test_ratio",
                    help="Test set proportion (unnormalized)")

    # ── stratification mode ───────────────────────────────────────────────────
    ap.add_argument(
        "--stratify",
        default="event",
        choices=["event", "track", "none"],
        help=(
            "Stratification key for the track-level split.\n"
            "  event    : by event label (cut_in / lane_change / lane_following)\n"
            "  track    : track-level random split (scenario label 미사용)\n"
            "  none     : random split on window indices (no track grouping)"
        ),
    )

    # ── eval stratification ───────────────────────────────────────────────────
    ap.add_argument(
        "--no_stratify_eval",
        action="store_true",
        default=False,
        help=(
            "If set, only train is stratified by scenario; "
            "val and test are split randomly from the remaining tracks."
        ),
    )

    # ── misc ──────────────────────────────────────────────────────────────────
    ap.add_argument("--seed", type=int, default=SEED, help="Random seed")

    return ap.parse_args()


def normalize_ratios(
    train: float, val: float, test: float,
) -> tuple[float, float, float]:
    if any(r <= 0 for r in [train, val, test]):
        raise ValueError("All split ratios must be positive.")
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        print(f"[WARNING] Ratios sum to {total:.4f} — auto-normalizing to 1.0.")
    return train / total, val / total, test / total


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    train_ratio, val_ratio, test_ratio = normalize_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )
    print(
        f"[INFO] Ratios   train={train_ratio:.3f}  "
        f"val={val_ratio:.3f}  test={test_ratio:.3f}"
    )
    print(f"[INFO] Stratify : {args.stratify}  "
          f"(eval={'random' if args.no_stratify_eval else 'stratified'})")

    # ── load data ─────────────────────────────────────────────────────────────
    if args.stratify in ("track", "none"):
        meta_dir = Path(args.meta_dir)
        rec_path = meta_dir / "meta_recordingId.npy"
        trk_path = meta_dir / "meta_trackId.npy"
        for p in (rec_path, trk_path):
            if not p.exists():
                raise FileNotFoundError(f"Meta file not found: {p}")
        df = pd.DataFrame({
            "recordingId": np.load(rec_path),
            "trackId":     np.load(trk_path),
        })
        df["global_index"] = df.index
        print(f"[INFO] Loaded {len(df):,} windows from: {meta_dir}")
    else:
        label_path = Path(args.label_file)
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")
        df = pd.read_csv(label_path)
        df["global_index"] = df.index
        print(f"[INFO] Loaded {len(df):,} windows from: {label_path}")

    # ── split ─────────────────────────────────────────────────────────────────
    if args.stratify == "none":
        train_idx, val_idx, test_idx = random_split(
            df, train_ratio, val_ratio, args.seed
        )
    elif args.stratify == "track":
        train_idx, val_idx, test_idx = track_random_split(
            df, train_ratio, val_ratio, args.seed
        )
    else:
        train_idx, val_idx, test_idx = scenario_split(
            df, args.stratify, train_ratio, val_ratio, args.seed,
            stratify_eval=not args.no_stratify_eval,
        )

    # ── label distribution summary ────────────────────────────────────────────
    print("\n[Label distribution per split]")
    print_label_dist(df, "Train", train_idx)
    print_label_dist(df, "Val  ", val_idx)
    print_label_dist(df, "Test ", test_idx)

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "val_indices.npy",   val_idx)
    np.save(out_dir / "test_indices.npy",  test_idx)

    total = len(train_idx) + len(val_idx) + len(test_idx)
    print(f"\n[OK] Saved splits → {out_dir.resolve()}")
    print(f"   Train : {len(train_idx):>8,}  ({len(train_idx)/total*100:.1f}%)")
    print(f"   Val   : {len(val_idx):>8,}  ({len(val_idx)/total*100:.1f}%)")
    print(f"   Test  : {len(test_idx):>8,}  ({len(test_idx)/total*100:.1f}%)")
    print(f"   Total : {total:>8,}")


if __name__ == "__main__":
    main()
