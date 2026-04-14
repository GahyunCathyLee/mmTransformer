"""
evaluate_mmT.py — mmTransformer evaluation  (config 단일 실행 방식)

실험 하나 = config 파일 하나.
cond / seed / 데이터 경로 / 체크포인트 위치는 모두 config 에서 읽습니다.

실행:
    python evaluate_mmT.py --config configs/c0_seed42.yaml
    python evaluate_mmT.py --config configs/c0_seed42.yaml --split val
    python evaluate_mmT.py --config configs/c0_seed42.yaml --measure_time
    python evaluate_mmT.py --config configs/c0_seed42.yaml --scenario_labels data/scenario_labels.csv
"""

import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.utils.utilities import load_config_data
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

# train.py 의 공통 객체 재사용
from train import (
    EXTRA_FEATURE_MAP,
    HighDDataset,
    build_model,
    set_seed,
)

TARGET_HZ = 3.0   # preprocess_mmT.py TARGET_FPS


# ─────────────────────────────────────────────────────────��────────────────────
# Scenario label helpers
# ────────────────────────────────────────────────────────────────────────────��─

def load_scenario_labels(path):
    path = Path(path)
    if not path.exists():
        print(f"[WARN] scenario_labels not found: {path}")
        return None
    df = pd.read_csv(path)
    required = {"recordingId", "trackId", "t0_frame"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] scenario_labels missing columns {missing}")
        return None
    has_event = "event_label" in df.columns
    has_state = "state_label" in df.columns
    if not has_event and not has_state:
        print("[WARN] scenario_labels has no event_label/state_label")
        return None
    lut = {}
    for row in df.itertuples(index=False):
        key = (int(row.recordingId), int(row.trackId), int(row.t0_frame))
        lut[key] = {
            "event_label": getattr(row, "event_label", None) if has_event else None,
            "state_label": getattr(row, "state_label", None) if has_state else None,
        }
    print(f"[INFO] Loaded scenario labels: {len(lut):,} entries from {path}")
    return lut


def _sep(widths, left="+", mid="+", right="+", fill="-"):
    return left + mid.join(fill * w for w in widths) + right


def print_scenario_results(stats, label_type):
    if not stats:
        return
    rows = sorted(stats.items(), key=lambda x: (x[0] == "unknown", x[0]))
    c_lbl = max(len(lbl) for lbl, _ in rows)
    c_lbl = max(c_lbl, len(label_type)) + 2
    c_n = 9
    c_m = 11
    ws = [c_lbl, c_n, c_m, c_m, c_m]

    print(f"\n====== Scenario Results [{label_type}] ======")
    print(_sep(ws))
    print(f"|{label_type:^{c_lbl}}|{'n':^{c_n}}"
          f"|{'ADE':^{c_m}}|{'FDE':^{c_m}}|{'RMSE':^{c_m}}|")
    print(_sep(ws))

    for lbl, (sa, sf, sr, n) in rows:
        if n == 0:
            continue
        print(f"|{lbl:^{c_lbl}}|{n:^{c_n},}"
              f"|{sa/n:^{c_m}.4f}|{sf/n:^{c_m}.4f}|{sr/n:^{c_m}.4f}|")
    print(_sep(ws))

    total_sa = sum(v[0] for v in stats.values())
    total_sf = sum(v[1] for v in stats.values())
    total_sr = sum(v[2] for v in stats.values())
    total_n = sum(v[3] for v in stats.values())
    N = max(1, total_n)
    print(f"|{'Total':^{c_lbl}}|{total_n:^{c_n},}"
          f"|{total_sa/N:^{c_m}.4f}|{total_sf/N:^{c_m}.4f}|{total_sr/N:^{c_m}.4f}|")
    print(_sep(ws))


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────��────
def compute_metrics_detailed(pred_trajs: torch.Tensor,
                              gt_trajs:  torch.Tensor) -> dict:
    """
    pred_trajs : [B, Q, T, 2]
    gt_trajs   : [B, T, 2]

    Returns dict of per-sample tensors:
        minADE, minFDE, RMSE, RMSE@1s … RMSE@5s
    """
    B, Q, T, _ = pred_trajs.shape

    # Best-of-Q by FDE
    dist_ep  = torch.norm(pred_trajs[:, :, -1, :] - gt_trajs[:, -1, :].unsqueeze(1), dim=-1)
    best_idx = torch.argmin(dist_ep, dim=-1)                # [B]
    best     = pred_trajs[torch.arange(B), best_idx]        # [B, T, 2]

    min_ade = torch.norm(pred_trajs - gt_trajs.unsqueeze(1), dim=-1).mean(-1).min(-1).values
    min_fde = dist_ep.min(-1).values

    sq_err       = torch.pow(best - gt_trajs, 2).sum(-1)    # [B, T]
    rmse_overall = torch.sqrt(sq_err.mean(-1))              # [B]

    rmse_at = {}
    for s in range(1, 6):
        step = min(int(s * TARGET_HZ) - 1, T - 1)
        rmse_at[f'RMSE@{s}s'] = torch.sqrt(sq_err[:, step])

    return {'minADE': min_ade, 'minFDE': min_fde,
            'RMSE': rmse_overall, **rmse_at}


# ──────────────────────────────────────────────────���───────────────────────────
# Inference latency
# ──────────────────────────────────────��───────────────────────────────────────
def measure_inference_time(model, dataset: HighDDataset,
                            device: torch.device, num_iters: int = 10_000):
    model.eval()
    sample = dataset[0]
    batch  = {k: v.unsqueeze(0).to(device)
              for k, v in sample.items() if isinstance(v, torch.Tensor)}

    latencies = []
    print(f"\nInference latency  ({num_iters:,} iters, batch=1)")
    with torch.no_grad():
        for _ in range(100):    # warm-up
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                model(batch)
        for _ in tqdm(range(num_iters), desc="Measuring"):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                model(batch)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    print(f"\n  avg={lat.mean():.3f} ms  min={lat.min():.3f} ms  "
          f"max={lat.max():.3f} ms  std={lat.std():.3f} ms\n")
    return {'avg': lat.mean(), 'min': lat.min(), 'max': lat.max(), 'std': lat.std()}


# ────────────────────────────────────────────────────────���─────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(cfg: dict, split: str, measure_time: bool, output_dir: Path,
             scenario_labels_path: str = None):
    exp_cfg   = cfg['exp']
    data_cfg  = cfg['data']
    train_cfg = cfg['train']

    cond         = exp_cfg['cond']
    feature_mode = exp_cfg['feature_mode']
    seed         = train_cfg['seed']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # ── 데이터 경로 ───────────────────────────────────────────────────────────
    h5_path  = data_cfg[split]['processed_data_path']
    map_path = data_cfg[split]['processed_maps_path']

    # ── 체크포인트 경로 ──
    ckpt_path = Path(train_cfg['ckpt_dir']) / cfg['_config_stem'] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  cond={cond}  feature_mode={feature_mode}  seed={seed}")
    print(f"  split={split}")
    print(f"  data={h5_path}")
    print(f"  ckpt={ckpt_path}")
    print(f"{'='*60}")

    # ── Scenario labels ──────────────────────────────────────────────��───────
    labels_lut = None
    if scenario_labels_path and not measure_time:
        labels_lut = load_scenario_labels(scenario_labels_path)

    # ── Dataset ─────────────────────────────���─────────────────────────────────
    need_meta = labels_lut is not None
    ds = HighDDataset(h5_path, map_path, return_meta=need_meta)

    batch_size  = data_cfg.get('batch_size', 512)
    num_workers = data_cfg.get('num_workers', 8)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ── Model ──────────────────────────���──────────────────────────────────────
    model = build_model(cfg, device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = (ckpt.get('model_state_dict')
             or ckpt.get('state_dict')
             or ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  체크포인트 로드 완료 (epoch={ckpt.get('epoch', '?')}  "
          f"val_rmse={ckpt.get('val_rmse', float('nan')):.4f})")

    # ── Inference time measurement ──────────────────────────────���─────────────
    if measure_time:
        measure_inference_time(model, ds, device)
        return

    model = torch.compile(model)

    # ── Evaluation loop ─────────────────────────────────��─────────────────────
    metric_keys = ['minADE', 'minFDE', 'RMSE'] + [f'RMSE@{s}s' for s in range(1, 6)]
    accum = {k: 0.0 for k in metric_keys}
    total = 0

    ev_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    st_stats = defaultdict(lambda: [0.0, 0.0, 0.0, 0])

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Evaluating [{split}]"):
            # Extract meta before moving tensors to device
            meta_recs   = batch.pop('META_REC', None)
            meta_tracks = batch.pop('META_TRACK', None)
            meta_frames = batch.pop('META_FRAME', None)

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                pred, _ = model(batch)

            target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred  # [B, Q, T, 2]
            target_gt   = batch['FUTURE'][:, 0, :, :2]                  # [B, T, 2]

            metrics = compute_metrics_detailed(target_pred, target_gt)
            B = target_gt.size(0)
            for k in metric_keys:
                accum[k] += metrics[k].sum().item()
            total += B

            # Per-scenario accumulation
            if labels_lut is not None and meta_recs is not None:
                ade_np  = metrics['minADE'].cpu().numpy()
                fde_np  = metrics['minFDE'].cpu().numpy()
                rmse_np = metrics['RMSE'].cpu().numpy()
                for i in range(B):
                    key = (int(meta_recs[i]), int(meta_tracks[i]), int(meta_frames[i]))
                    lab = labels_lut.get(key)
                    if lab is None:
                        continue
                    ev = lab.get("event_label") or "unknown"
                    st = lab.get("state_label") or "unknown"
                    for acc, lbl in ((ev_stats, ev), (st_stats, st)):
                        acc[lbl][0] += float(ade_np[i])
                        acc[lbl][1] += float(fde_np[i])
                        acc[lbl][2] += float(rmse_np[i])
                        acc[lbl][3] += 1

    final = {k: accum[k] / total for k in metric_keys}

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n  ── 평가 결과 ({cond} / seed={seed} / {split}) ──")
    print(f"  Total samples : {total:,}")
    print(f"  minADE  = {final['minADE']:.4f} m")
    print(f"  minFDE  = {final['minFDE']:.4f} m")
    print(f"  RMSE    = {final['RMSE']:.4f} m")
    print(f"  RMSE@t  : " + "  ".join(
        f"{s}s={final[f'RMSE@{s}s']:.4f}" for s in range(1, 6)))

    # ── Scenario results ──────────────────────���───────────────────────────────
    if ev_stats:
        print_scenario_results(ev_stats, label_type="Event")
    if st_stats:
        print_scenario_results(st_stats, label_type="State")

    # ── CSV 저장 ────────────────────────────────��─────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{cfg['_config_stem']}_{split}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f,
                                fieldnames=['cond', 'seed', 'split'] + metric_keys)
        writer.writeheader()
        writer.writerow({'cond': cond, 'seed': seed, 'split': split, **final})
    print(f"\n  CSV → {csv_path}")

    return final


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────���───────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       type=str, required=True,
                        help='실험 config yaml 경로 (예: configs/c0_seed42.yaml)')
    parser.add_argument('--split',        type=str, default='test',
                        choices=['val', 'test'],
                        help='평가할 데이터 split (default: test)')
    parser.add_argument('--measure_time', action='store_true',
                        help='추론 시간 측정 모드 (batch=1, 10k iters)')
    parser.add_argument('--output_dir',   type=str, default='results',
                        help='CSV 결과 저장 디렉토리')
    parser.add_argument('--scenario_labels', type=str, default=None,
                        help='Path to scenario_labels.csv for per-scenario breakdown')
    args = parser.parse_args()

    cfg = load_config_data(args.config)
    cfg['_config_stem'] = Path(args.config).stem   # e.g. "c0_seed42"
    evaluate(cfg, args.split, args.measure_time, Path(args.output_dir),
             args.scenario_labels)
