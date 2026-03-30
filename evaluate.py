"""
evaluate_mmT.py — mmTransformer evaluation  (config 단일 실행 방식)

실험 하나 = config 파일 하나.
cond / seed / 데이터 경로 / 체크포인트 위치는 모두 config 에서 읽습니다.

실행:
    python evaluate_mmT.py --config configs/c0_seed42.yaml
    python evaluate_mmT.py --config configs/c0_seed42.yaml --split val
    python evaluate_mmT.py --config configs/c0_seed42.yaml --measure_time
"""

import argparse
import csv
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

# train.py 의 공통 객체 재사용
from train import (
    EXTRA_FEATURE_MAP,
    HighDDataset,
    build_model,
    set_seed,
)

TARGET_HZ = 3.0   # preprocess_mmT.py TARGET_FPS


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# Inference latency
# ──────────────────────────────────────────────────────────────────────────────
def measure_inference_time(model, dataset: HighDDataset,
                            device: torch.device, num_iters: int = 10_000):
    model.eval()
    sample = dataset[0]
    batch  = {k: v.unsqueeze(0).to(device)
              for k, v in sample.items() if isinstance(v, torch.Tensor)}

    latencies = []
    print(f"\n⏱  Inference latency  ({num_iters:,} iters, batch=1)")
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


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(cfg: dict, split: str, measure_time: bool, output_dir: Path):
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

    # ── 체크포인트 경로  (train_mmT.py 와 동일한 규칙: ckpt_dir / config_stem) ──
    ckpt_path = Path(train_cfg['ckpt_dir']) / cfg['_config_stem'] / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  cond={cond}  feature_mode={feature_mode}  seed={seed}")
    print(f"  split={split}")
    print(f"  data={h5_path}")
    print(f"  ckpt={ckpt_path}")
    print(f"{'='*60}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = HighDDataset(h5_path, map_path)

    batch_size  = data_cfg.get('batch_size', 512)
    num_workers = data_cfg.get('num_workers', 8)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = (ckpt.get('model_state_dict')
             or ckpt.get('state_dict')
             or ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  체크포인트 로드 완료 (epoch={ckpt.get('epoch', '?')}  "
          f"val_rmse={ckpt.get('val_rmse', float('nan')):.4f})")

    # ── Inference time measurement ────────────────────────────────────────────
    if measure_time:
        measure_inference_time(model, ds, device)
        return

    model = torch.compile(model)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    metric_keys = ['minADE', 'minFDE', 'RMSE'] + [f'RMSE@{s}s' for s in range(1, 6)]
    accum = {k: 0.0 for k in metric_keys}
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Evaluating [{split}]"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                pred, _ = model(batch)

            target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred  # [B, Q, T, 2]
            target_gt   = batch['FUTURE'][:, 0, :, :2]                  # [B, T, 2]

            metrics = compute_metrics_detailed(target_pred, target_gt)
            for k in metric_keys:
                accum[k] += metrics[k].sum().item()
            total += target_gt.size(0)

    final = {k: accum[k] / total for k in metric_keys}

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print(f"\n  ── 평가 결과 ({cond} / seed={seed} / {split}) ──")
    print(f"  Total samples : {total:,}")
    print(f"  minADE  = {final['minADE']:.4f} m")
    print(f"  minFDE  = {final['minFDE']:.4f} m")
    print(f"  RMSE    = {final['RMSE']:.4f} m")
    print(f"  RMSE@t  : " + "  ".join(
        f"{s}s={final[f'RMSE@{s}s']:.4f}" for s in range(1, 6)))

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
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
# ──────────────────────────────────────────────────────────────────────────────
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
    args = parser.parse_args()

    cfg = load_config_data(args.config)
    cfg['_config_stem'] = Path(args.config).stem   # e.g. "c0_seed42"
    evaluate(cfg, args.split, args.measure_time, Path(args.output_dir))