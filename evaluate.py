"""
evaluate_mmT.py — mmTransformer evaluation on neighformer mmap datasets

Loads the best checkpoint for each (condition, seed) pair produced by train_mmT.py
and reports detailed metrics:  minADE, minFDE, RMSE, RMSE@1s … RMSE@5s

Usage examples
──────────────
  # Evaluate all conditions, all seeds, on test split
  python evaluate_mmT.py --config configs/baseline.yaml

  # Single condition + seed
  python evaluate_mmT.py --config configs/baseline.yaml --cond c1 --seed 42

  # Measure inference latency (batch=1, 10k iters)
  python evaluate_mmT.py --config configs/baseline.yaml --cond c0 --seed 42 --measure_time
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.utils.utilities import load_config_data
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

# ── reuse shared constants / classes from train_mmT ──────────────────────────
from train_mmT import (
    CONDITIONS, SEEDS,
    T_HIST, T_FUT, K_NB, MAX_AGENTS, MAX_LANES, HIST_CHANNELS,
    NeighformerMmapDataset,
    build_model,
    set_seed,
)

# target_hz for neighformer mmap
TARGET_HZ = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics_detailed(pred_trajs: torch.Tensor,
                              gt_trajs:  torch.Tensor,
                              fps: float = TARGET_HZ) -> dict:
    """
    pred_trajs : [B, K, T, 2]
    gt_trajs   : [B, T, 2]

    Returns dict of tensors (one value per sample):
        minADE, minFDE, RMSE, RMSE@1s … RMSE@5s
    """
    B, K, T, _ = pred_trajs.shape

    # Best-of-K by FDE
    pred_ep = pred_trajs[:, :, -1, :]                            # [B, K, 2]
    gt_ep   = gt_trajs[:, -1, :].unsqueeze(1)                    # [B, 1, 2]
    dist_ep = torch.norm(pred_ep - gt_ep, dim=-1)                # [B, K]
    best_idx = torch.argmin(dist_ep, dim=-1)                     # [B]
    best_traj = pred_trajs[torch.arange(B), best_idx]            # [B, T, 2]

    # minADE
    dist_all = torch.norm(pred_trajs - gt_trajs.unsqueeze(1), dim=-1)  # [B, K, T]
    min_ade  = dist_all.mean(dim=-1).min(dim=-1).values                # [B]

    # minFDE
    min_fde  = dist_ep.min(dim=-1).values                         # [B]

    # RMSE (full horizon, best-of-K)
    sq_err       = torch.pow(best_traj - gt_trajs, 2).sum(dim=-1) # [B, T]
    rmse_overall = torch.sqrt(sq_err.mean(dim=-1))                 # [B]

    # RMSE@Ns  (exact timestep, 1 s … 5 s)
    rmse_at = {}
    for s in range(1, 6):
        step_idx = min(int(s * fps) - 1, T - 1)
        rmse_at[f'RMSE@{s}s'] = torch.sqrt(sq_err[:, step_idx])   # [B]

    return {'minADE': min_ade, 'minFDE': min_fde,
            'RMSE': rmse_overall, **rmse_at}


# ──────────────────────────────────────────────────────────────────────────────
# Inference time measurement
# ──────────────────────────────────────────────────────────────────────────────
def measure_inference_time(model, dataset: NeighformerMmapDataset,
                            device: torch.device, num_iters: int = 10_000):
    model.eval()
    sample = dataset[0]
    batch  = {k: v.unsqueeze(0).to(device)
              for k, v in sample.items() if isinstance(v, torch.Tensor)}

    latencies = []
    print(f"\n⏱ Inference latency  ({num_iters:,} iters, batch=1)")

    with torch.no_grad():
        for _ in range(100):                        # warm-up
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
    return {'avg': lat.mean(), 'min': lat.min(),
            'max': lat.max(), 'std': lat.std()}


# ──────────────────────────────────────────────────────────────────────────────
# Single evaluation run  (one condition × one seed × one split)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_one(
    cond_name:   str,
    mmap_dir:    Path,
    seed:        int,
    split:       str,
    cfg:         dict,
    device:      torch.device,
    ckpt_path:   Path,
    batch_size:  int,
    num_workers: int,
    measure_time: bool = False,
) -> dict:
    set_seed(seed)

    ds     = NeighformerMmapDataset(mmap_dir / split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = build_model(cfg, device)
    print(f"\n  Loading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = (ckpt.get('model_state_dict')
             or ckpt.get('state_dict')
             or ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    if measure_time:
        measure_inference_time(model, ds, device)
        return {}

    model = torch.compile(model)

    metric_keys = ['minADE', 'minFDE', 'RMSE'] + [f'RMSE@{s}s' for s in range(1, 6)]
    accum = {k: 0.0 for k in metric_keys}
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Eval [{cond_name}/seed{seed}/{split}]"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                pred, _ = model(batch)

            target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred   # [B, Q, T, 2]
            target_gt   = batch['FUTURE'][:, 0, :, :2]                    # [B, T, 2]

            metrics = compute_metrics_detailed(target_pred, target_gt)
            for k in metric_keys:
                accum[k] += metrics[k].sum().item()
            total += target_gt.size(0)

    final = {k: accum[k] / total for k in metric_keys}

    # Print
    print(f"\n  ── Result: {cond_name} | seed={seed} | split={split} ──")
    print(f"  minADE = {final['minADE']:.4f} m")
    print(f"  minFDE = {final['minFDE']:.4f} m")
    print(f"  RMSE   = {final['RMSE']:.4f} m")
    print(f"  RMSE@t :  " + "  ".join(
        f"{s}s={final[f'RMSE@{s}s']:.4f}" for s in range(1, 6)))

    return final


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',       type=str, default='configs/baseline.yaml')
    p.add_argument('--data_dir',     type=str, default='data/highD')
    p.add_argument('--ckpt_dir',     type=str, default='checkpoints',
                   help='Root checkpoint dir (must match --save_dir used in train_mmT.py)')
    p.add_argument('--output_dir',   type=str, default='results')
    p.add_argument('--split',        type=str, default='test', choices=['val', 'test'])
    p.add_argument('--cond',         type=str, default=None,
                   choices=list(CONDITIONS.keys()) + [None])
    p.add_argument('--seed',         type=int, default=None)
    p.add_argument('--measure_time', action='store_true')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_config_data(args.config)

    data_dir    = Path(args.data_dir)
    ckpt_root   = Path(args.ckpt_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond_names = [args.cond] if args.cond else list(CONDITIONS.keys())
    seeds      = [args.seed] if args.seed is not None else SEEDS

    batch_size  = cfg.get('data', {}).get('batch_size', 512)
    num_workers = cfg.get('data', {}).get('num_workers', 8)

    metric_keys = ['minADE', 'minFDE', 'RMSE'] + [f'RMSE@{s}s' for s in range(1, 6)]

    # results[cond][seed] = {metric: value, ...}
    results: dict = {c: {} for c in cond_names}
    csv_rows = []

    for cond_name in cond_names:
        mmap_dir = data_dir / CONDITIONS[cond_name]["mmap_subdir"]
        print(f"\n{'#'*60}")
        print(f"# Condition : {cond_name}  — {CONDITIONS[cond_name]['desc']}")
        print(f"{'#'*60}")

        for seed in seeds:
            ckpt_path = ckpt_root / cond_name / f"seed{seed}" / "best.pt"
            if not ckpt_path.exists():
                print(f"  [WARN] Checkpoint not found: {ckpt_path} — skipping")
                continue

            final = evaluate_one(
                cond_name    = cond_name,
                mmap_dir     = mmap_dir,
                seed         = seed,
                split        = args.split,
                cfg          = cfg,
                device       = device,
                ckpt_path    = ckpt_path,
                batch_size   = batch_size,
                num_workers  = num_workers,
                measure_time = args.measure_time,
            )
            if not final:
                continue

            results[cond_name][seed] = final
            csv_rows.append({'cond': cond_name, 'seed': seed, **final})

    if args.measure_time:
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  EVALUATION SUMMARY  [{args.split}]")
    print(f"{'='*80}")

    col_w = 10
    header_cols = ['cond', 'seed'] + metric_keys
    header_str  = '  '.join(f"{c:>{col_w}}" for c in header_cols)
    print(header_str)
    print('-' * len(header_str))

    for cond_name in cond_names:
        seed_vals: dict = {k: [] for k in metric_keys}

        for seed in seeds:
            m = results[cond_name].get(seed)
            if m is None:
                continue
            row = {'cond': cond_name, 'seed': seed}
            for k in metric_keys:
                row[k] = m[k]
                seed_vals[k].append(m[k])
            print('  '.join(f"{str(row[c]):>{col_w}}" if c in ('cond', 'seed')
                             else f"{row[c]:>{col_w}.4f}"
                             for c in header_cols))

        if not any(seed_vals.values()):
            continue

        # mean / std rows
        for stat, fn in [('mean', np.nanmean), ('std', np.nanstd)]:
            stat_row = {'cond': cond_name, 'seed': stat}
            for k in metric_keys:
                stat_row[k] = fn(seed_vals[k]) if seed_vals[k] else float('nan')
            print('  '.join(f"{str(stat_row[c]):>{col_w}}" if c in ('cond', 'seed')
                             else f"{stat_row[c]:>{col_w}.4f}"
                             for c in header_cols))
            csv_rows.append({'cond': cond_name, 'seed': stat, **{k: stat_row[k] for k in metric_keys}})
        print()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = output_dir / f"eval_{args.split}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cond', 'seed'] + metric_keys)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Summary CSV saved → {csv_path}")


if __name__ == '__main__':
    main()