"""
train_mmT.py — mmTransformer training on neighformer mmap datasets

Experiment conditions
─────────────────────
  c0 : preprocess_neighformer.py --history_sec 2
       (mmap_dir default: data/highD/mmap)

  c1 : preprocess_neighformer.py --history_sec 2 --lc_version v4
          --mmap_dir v4_7_g2 --gate_topn 2 --lis_mode 7
       (mmap_dir: data/highD/v4_7_g2)

  c2 : preprocess_neighformer.py --history_sec 2 --lc_version v4
          --mmap_dir v4_7_g2_slot3 --gate_topn 2 --lis_mode 7 --slotImportance 0.3
       (mmap_dir: data/highD/v4_7_g2_slot3)

Each condition is trained 5 times with seeds [42, 1234, 3407, 0, 777].

mmap file layout (per condition dir)
──────────────────────────────────
  x_ego.npy      (N, T,    6)   ego history  [x,y,xV,yV,xA,yA]
  x_nb.npy       (N, T, K, 13) neighbor features (13-ch)
  nb_mask.npy    (N, T, K)     bool existence mask
  y.npy          (N, Tf,   2)  future [x,y]
  y_vel.npy      (N, Tf,   2)  future velocity
  y_acc.npy      (N, Tf,   2)  future acceleration
  x_last_abs.npy (N, 2)        absolute ref position

mmTransformer HISTORY input shape: [B, A, T, C]
  slot 0 = ego  → built from x_ego
  slots 1–K = neighbors → built from x_nb
  C = 4 + num_extra    (x, y, timestamp, mask, [extras])

POS input shape: [B, A, 2]
  relative position of each agent at the last history step

VALID_LEN: [B, 2]  (num_valid_agents, num_valid_lanes=0 — lanes not used here)
"""

import os
import csv
import random
import argparse
import warnings
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from lib.utils.utilities import load_config_data
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

# ──────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ──────────────────────────────────────────────────────────────────────────────
CONDITIONS = {
    "c0": {
        "mmap_subdir": "mmap",           # relative to data_dir
        "desc": "baseline (history_sec=2)",
    },
    "c1": {
        "mmap_subdir": "v4_7_g2",
        "desc": "lc_version=v4, lis_mode=7, gate_topn=2",
    },
    "c2": {
        "mmap_subdir": "v4_7_g2_slot3",
        "desc": "lc_version=v4, lis_mode=7, gate_topn=2, slotImportance=0.3",
    },
}

SEEDS = [42, 1234, 3407, 0, 777]

# neighformer mmap constants
# history_sec=2, target_hz=3  → T=6
# future_sec=5,  target_hz=3  → Tf=15
T_HIST   = 6
T_FUT    = 15
K_NB     = 8    # number of neighbor slots
EGO_DIM  = 6    # x_ego channels
NB_DIM   = 13   # x_nb channels per slot
MAX_AGENTS = K_NB + 1   # ego + 8 neighbors = 9  (matches mmT MAX_AGENTS)
MAX_LANES  = 0           # lane data not used (no lane map in mmap format)

# Extra feature selection from the 13-ch neighbor vector
# [0:dx, 1:dy, 2:dvx, 3:dvy, 4:dax, 5:day, 6:lc_state, 7:lit, 8:lis,
#  9:gate, 10:I_x, 11:I_y, 12:I]
EXTRA_NB_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # all 13 → slice below
# Channels fed into mmT HISTORY tensor per agent:
#   [x, y, timestamp, mask]  +  extra_nb  →  4 + len(EXTRA_NB_INDICES)
# For ego: x,y from x_ego; extra = zeros (ego has no relative neighbor features)
HIST_CHANNELS = 4 + len(EXTRA_NB_INDICES)   # = 17


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def build_timestamps(T: int, hz: float) -> np.ndarray:
    """Relative timestamps: [..., -2/hz, -1/hz, 0]  shape (T,)"""
    return np.arange(-(T - 1), 1, dtype=np.float32) / hz


TIMESTAMPS = build_timestamps(T_HIST, hz=3.0)   # shape (6,)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class NeighformerMmapDataset(Dataset):
    """
    Reads neighformer mmap npy files and converts them into the
    mmTransformer HISTORY / FUTURE / POS / VALID_LEN format.

    HISTORY tensor layout  [A, T, C]  where C = 4 + 13 = 17
      dim 0 : position x (ego-relative, rotated)  ← x_ego[:,0] or x_nb dx
      dim 1 : position y                           ← x_ego[:,1] or x_nb dy
      dim 2 : timestamp
      dim 3 : existence mask (1 = present)
      dim 4–16 : extra neighbor features [dx,dy,dvx,dvy,dax,day,lc,lit,lis,gate,Ix,Iy,I]
                 (0 for ego slot)

    POS tensor  [A, 2]  : position at last history step (t=T-1)
      ego : x_ego[-1, :2]
      nb  : x_nb[-1, k, :2]  (dx, dy at last step)

    FUTURE tensor  [A, Tf, 3]
      ego : y[:, :2], mask=1
      nb  : zeros (mmT only predicts ego)

    VALID_LEN  [2] : [num_valid_agents, 0]
      num_valid_agents = 1 (ego) + number of slots with at least one present frame
    """

    def __init__(self, mmap_dir: Path):
        self.mmap_dir = Path(mmap_dir)
        self._load()

    def _load(self):
        d = self.mmap_dir
        # memory-map in read-only mode  (zero-copy, shared across workers)
        self.x_ego      = np.load(d / "x_ego.npy",      mmap_mode='r')   # (N, T, 6)
        self.x_nb       = np.load(d / "x_nb.npy",       mmap_mode='r')   # (N, T, K, 13)
        self.nb_mask    = np.load(d / "nb_mask.npy",     mmap_mode='r')   # (N, T, K) bool
        self.y          = np.load(d / "y.npy",           mmap_mode='r')   # (N, Tf, 2)
        self.x_last_abs = np.load(d / "x_last_abs.npy", mmap_mode='r')   # (N, 2)
        self.N = self.x_ego.shape[0]
        print(f"  [Dataset] {d}  —  {self.N:,} samples")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # ── raw mmap slices (copies to numpy arrays) ──────────────────────────
        ego = self.x_ego[idx].copy()        # (T, 6)
        nb  = self.x_nb[idx].copy()         # (T, K, 13)
        nm  = self.nb_mask[idx].copy()      # (T, K) bool
        fut = self.y[idx].copy()            # (Tf, 2)

        T  = ego.shape[0]    # T_HIST
        Tf = fut.shape[0]    # T_FUT

        # ── HISTORY  [A, T, C] ────────────────────────────────────────────────
        hist = np.zeros((MAX_AGENTS, T, HIST_CHANNELS), dtype=np.float32)

        # slot 0: ego
        #   x, y  ← ego_x - ego_x_last, ego_y - ego_y_last  (already shifted by neighformer)
        hist[0, :, 0] = ego[:, 0]          # x  (ego-centric, shifted to last pos = origin)
        hist[0, :, 1] = ego[:, 1]          # y
        hist[0, :, 2] = TIMESTAMPS         # timestamp
        hist[0, :, 3] = 1.0               # mask: ego always present
        # dims 4..16 → 0 (ego has no relative neighbor features)

        # slots 1..K: neighbors
        any_present = np.zeros(K_NB, dtype=bool)
        for k in range(K_NB):
            present = nm[:, k]             # (T,) bool
            any_present[k] = present.any()
            if not any_present[k]:
                continue
            a = k + 1
            hist[a, :, 0]  = nb[:, k, 0]  # dx
            hist[a, :, 1]  = nb[:, k, 1]  # dy
            hist[a, :, 2]  = TIMESTAMPS
            hist[a, :, 3]  = present.astype(np.float32)
            hist[a, :, 4:] = nb[:, k, :]  # all 13 extra channels

        # ── POS  [A, 2] ───────────────────────────────────────────────────────
        pos = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
        pos[0, 0] = ego[-1, 0]
        pos[0, 1] = ego[-1, 1]
        for k in range(K_NB):
            if any_present[k]:
                pos[k + 1] = nb[-1, k, 0:2]

        # ── FUTURE  [A, Tf, 3] ────────────────────────────────────────────────
        future = np.zeros((MAX_AGENTS, Tf, 3), dtype=np.float32)
        future[0, :, :2] = fut
        future[0, :, 2]  = 1.0   # ego future always valid

        # ── VALID_LEN  [2] ────────────────────────────────────────────────────
        n_valid_agents = int(1 + any_present.sum())
        valid_len = np.array([n_valid_agents, 0], dtype=np.int64)

        return {
            'HISTORY':   torch.from_numpy(hist),
            'FUTURE':    torch.from_numpy(future),
            'POS':       torch.from_numpy(pos),
            'VALID_LEN': torch.from_numpy(valid_len),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_loss(pred_trajs, pred_confs, gt_trajs):
    """
    pred_trajs : [B, K, T, 2]
    pred_confs : [B, K]
    gt_trajs   : [B, T, 2]
    """
    B, K, T, _ = pred_trajs.shape

    pred_endpoints = pred_trajs[:, :, -1, :]
    gt_endpoints   = gt_trajs[:, -1, :].unsqueeze(1)
    distances      = torch.norm(pred_endpoints - gt_endpoints, dim=-1)
    best_k_idx     = torch.argmin(distances, dim=-1)

    best_pred = pred_trajs[torch.arange(B), best_k_idx]
    loss_reg  = F.smooth_l1_loss(best_pred, gt_trajs)

    if pred_confs.min() >= 0 and pred_confs.max() <= 1:
        loss_cls = F.nll_loss(torch.log(pred_confs + 1e-9), best_k_idx)
    else:
        loss_cls = F.cross_entropy(pred_confs, best_k_idx)

    return loss_reg + loss_cls, loss_reg, loss_cls, best_k_idx


def compute_metrics(pred_traj, gt_traj):
    """
    pred_traj : [B, T, 2]  (best-of-K selected)
    gt_traj   : [B, T, 2]
    Returns scalar ADE, RMSE (mean over batch)
    """
    dist = torch.norm(pred_traj - gt_traj, dim=-1)   # [B, T]
    ade  = dist.mean().item()
    mse  = torch.pow(pred_traj - gt_traj, 2).sum(dim=-1).mean(dim=-1)
    rmse = torch.sqrt(mse).mean().item()
    return ade, rmse


# ──────────────────────────────────────────────────────────────────────────────
# Build model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(cfg: dict, device: torch.device) -> mmTrans:
    model_cfg = dict(cfg.get('model', {}))
    model_cfg['in_channels']   = HIST_CHANNELS
    model_cfg['max_lane_num']  = MAX_LANES
    model_cfg['max_agent_num'] = MAX_AGENTS
    model_cfg['lane_channels'] = 7                    # unused but mmTrans expects it
    future_frames              = model_cfg.get('future_num_frames', T_FUT)
    model_cfg['future_num_frames'] = future_frames
    model_cfg['out_channels']  = future_frames * 2
    return mmTrans(STF, model_cfg).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Single training run  (one condition × one seed)
# ──────────────────────────────────────────────────────────────────────────────
def run_one(
    cond_name: str,
    mmap_dir: Path,
    seed: int,
    cfg: dict,
    device: torch.device,
    base_save_dir: Path,
    base_log_dir: Path,
) -> dict:
    """Train and return best val metrics."""
    set_seed(seed)

    trial_tag  = f"{cond_name}_seed{seed}"
    save_dir   = base_save_dir / cond_name / f"seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path  = save_dir / "best.pt"

    log_dir = base_log_dir / trial_tag / datetime.now().strftime("%m%d-%H%M")
    writer  = SummaryWriter(log_dir=str(log_dir))

    print(f"\n{'='*60}")
    print(f"  Condition : {cond_name}  |  Seed : {seed}")
    print(f"  mmap_dir  : {mmap_dir}")
    print(f"  save_dir  : {save_dir}")
    print(f"{'='*60}")

    train_ds = NeighformerMmapDataset(mmap_dir / "train")
    val_ds   = NeighformerMmapDataset(mmap_dir / "val")

    batch_size         = cfg.get('data', {}).get('batch_size', 512)
    num_workers        = cfg.get('data', {}).get('num_workers', 8)
    persistent_workers = cfg.get('train', {}).get('persistent_workers', True)
    num_epochs         = cfg.get('train', {}).get('epochs', 200)
    use_amp            = cfg.get('train', {}).get('use_amp', True)
    lr                 = float(cfg.get('train', {}).get('lr', 1e-4))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=persistent_workers,
        worker_init_fn=lambda wid: np.random.seed(seed + wid),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent_workers,
    )

    raw_model = build_model(cfg, device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr)
    scaler    = GradScaler('cuda') if use_amp else None

    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100,
    )

    model      = torch.compile(raw_model)
    best_rmse  = float('inf')
    best_metrics = {}

    for epoch in range(1, num_epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss = t_ade = t_rmse = 0.0
        pbar = tqdm(train_loader, desc=f"[{trial_tag}] Epoch {epoch}/{num_epochs} Train",
                    leave=False)

        for bi, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = autocast(device_type='cuda') if use_amp else nullcontext()

            with ctx:
                pred, conf = model(batch)
                # pred: [B, A, Q, T, 2]  or [B, Q, T, 2] — take ego slot 0
                target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
                target_conf = conf[:, 0, ...] if conf.dim() == 3 else conf
                target_gt   = batch['FUTURE'][:, 0, :, :2]    # ego future
                loss, l_reg, l_cls, bki = compute_loss(target_pred, target_conf, target_gt)

                if torch.isnan(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            global_step = (epoch - 1) * len(train_loader) + bi
            writer.add_scalar('Train/Loss',  loss.item(), global_step)
            writer.add_scalar('Train/LR',    optimizer.param_groups[0]['lr'], global_step)

            with torch.no_grad():
                best_traj = target_pred[torch.arange(target_pred.size(0)), bki]
                ade, rmse = compute_metrics(best_traj, target_gt)
            t_loss += loss.item(); t_ade += ade; t_rmse += rmse
            pbar.set_postfix({'ade': f'{ade:.3f}', 'rmse': f'{rmse:.3f}'})

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss = v_ade = v_rmse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device, non_blocking=True)
                ctx = autocast(device_type='cuda') if use_amp else nullcontext()
                with ctx:
                    pred, conf = model(batch)
                    target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
                    target_conf = conf[:, 0, ...] if conf.dim() == 3 else conf
                    target_gt   = batch['FUTURE'][:, 0, :, :2]
                    loss, _, _, bki = compute_loss(target_pred, target_conf, target_gt)
                best_traj = target_pred[torch.arange(target_pred.size(0)), bki]
                ade, rmse = compute_metrics(best_traj, target_gt)
                v_loss += loss.item(); v_ade += ade; v_rmse += rmse

        n_val = len(val_loader)
        avg_v_loss = v_loss / n_val
        avg_v_ade  = v_ade  / n_val
        avg_v_rmse = v_rmse / n_val

        writer.add_scalar('Loss/Train_Epoch', t_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Val_Epoch',   avg_v_loss, epoch)
        writer.add_scalar('Metric/Val_ADE',   avg_v_ade,  epoch)
        writer.add_scalar('Metric/Val_RMSE',  avg_v_rmse, epoch)

        print(f"  [{trial_tag}] Epoch {epoch:3d} | "
              f"val_loss={avg_v_loss:.4f}  ADE={avg_v_ade:.4f}  RMSE={avg_v_rmse:.4f}")

        if avg_v_rmse < best_rmse:
            best_rmse = avg_v_rmse
            best_metrics = {'val_ade': avg_v_ade, 'val_rmse': avg_v_rmse}
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse':           best_rmse,
                'val_ade':            avg_v_ade,
                'cond':               cond_name,
                'seed':               seed,
            }, best_path)
            print(f"  ⭐ Best updated → RMSE={best_rmse:.4f}  [{best_path}]")

    writer.close()
    return best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',    type=str, default='configs/baseline.yaml')
    p.add_argument('--data_dir',  type=str, default='data/highD',
                   help='Root data dir that contains mmap subdirs for each condition')
    p.add_argument('--save_dir',  type=str, default='checkpoints')
    p.add_argument('--log_dir',   type=str, default='logs')
    p.add_argument('--cond',      type=str, default=None,
                   choices=list(CONDITIONS.keys()) + [None],
                   help='Run a single condition only (default: run all c0/c1/c2)')
    p.add_argument('--seed',      type=int, default=None,
                   help='Run a single seed only (default: run all 5 seeds)')
    p.add_argument('--resume',    action='store_true',
                   help='Skip if best.pt already exists for that trial')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = load_config_data(args.config)

    data_dir     = Path(args.data_dir)
    base_save    = Path(args.save_dir)
    base_log     = Path(args.log_dir)

    cond_names = [args.cond] if args.cond else list(CONDITIONS.keys())
    seeds      = [args.seed] if args.seed is not None else SEEDS

    # Summary table
    # results[cond][seed] = {'val_ade': ..., 'val_rmse': ...}
    results: dict = {c: {} for c in cond_names}

    for cond_name in cond_names:
        mmap_dir = data_dir / CONDITIONS[cond_name]["mmap_subdir"]
        print(f"\n{'#'*60}")
        print(f"# Condition : {cond_name}  — {CONDITIONS[cond_name]['desc']}")
        print(f"# mmap_dir  : {mmap_dir}")
        print(f"{'#'*60}")

        for seed in seeds:
            best_path = base_save / cond_name / f"seed{seed}" / "best.pt"
            if args.resume and best_path.exists():
                # Load saved metrics for the summary table
                ckpt = torch.load(best_path, map_location='cpu')
                results[cond_name][seed] = {
                    'val_ade':  ckpt.get('val_ade',  float('nan')),
                    'val_rmse': ckpt.get('val_rmse', float('nan')),
                }
                print(f"  [Skip — already done] {cond_name}/seed{seed}  "
                      f"RMSE={results[cond_name][seed]['val_rmse']:.4f}")
                continue

            metrics = run_one(
                cond_name  = cond_name,
                mmap_dir   = mmap_dir,
                seed       = seed,
                cfg        = cfg,
                device     = device,
                base_save_dir = base_save,
                base_log_dir  = base_log,
            )
            results[cond_name][seed] = metrics

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    header = f"{'Condition':<10} {'Seed':>8}  {'val_ADE':>10}  {'val_RMSE':>10}"
    print(header)
    print('-' * len(header))

    csv_rows = []
    for cond_name in cond_names:
        ades  = []
        rmses = []
        for seed in seeds:
            m = results[cond_name].get(seed, {})
            ade_v  = m.get('val_ade',  float('nan'))
            rmse_v = m.get('val_rmse', float('nan'))
            ades.append(ade_v); rmses.append(rmse_v)
            row_str = f"  {cond_name:<10} {seed:>8}  {ade_v:>10.4f}  {rmse_v:>10.4f}"
            print(row_str)
            csv_rows.append({'cond': cond_name, 'seed': seed,
                             'val_ade': ade_v, 'val_rmse': rmse_v})

        mean_ade  = np.nanmean(ades)
        mean_rmse = np.nanmean(rmses)
        std_ade   = np.nanstd(ades)
        std_rmse  = np.nanstd(rmses)
        print(f"  {cond_name:<10} {'mean':>8}  {mean_ade:>10.4f}  {mean_rmse:>10.4f}")
        print(f"  {cond_name:<10} {'std':>8}  {std_ade:>10.4f}  {std_rmse:>10.4f}")
        print()
        csv_rows.append({'cond': cond_name, 'seed': 'mean',
                         'val_ade': mean_ade, 'val_rmse': mean_rmse})
        csv_rows.append({'cond': cond_name, 'seed': 'std',
                         'val_ade': std_ade,  'val_rmse': std_rmse})

    # Save summary CSV
    out_csv = base_save / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cond', 'seed', 'val_ade', 'val_rmse'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Summary saved → {out_csv}")


if __name__ == '__main__':
    main()