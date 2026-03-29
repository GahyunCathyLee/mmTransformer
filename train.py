"""
train_mmT.py — mmTransformer  (config 단일 실행 방식)

실험 하나 = config 파일 하나.
cond / seed / 데이터 경로는 모두 config 에서 읽습니다.

실행:
    python train_mmT.py --config configs/c0_seed42.yaml
    python train_mmT.py --config configs/c0_seed1234.yaml
    ...
"""

import argparse
import random
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pickle
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
# Feature map  (preprocess_mmT.py 의 EXTRA_FEATURE_MAP 과 동일해야 함)
# HISTORY 채널 수 = 4(x, y, timestamp, mask) + len(extra_indices)
# ──────────────────────────────────────────────────────────────────────────────
EXTRA_FEATURE_MAP = {
    'baseline':   [0, 1, 2, 3, 4, 5],      # 10 ch
    'importance': [0, 1, 2, 3, 4, 5, 12],  # 11 ch
    'Iy':         [0, 1, 2, 3, 4, 5, 11],  # 11 ch
}


# ──────────────────────────────────────────────────────────────────────────────
# Seed
# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class HighDDataset(Dataset):
    def __init__(self, h5_path: str, map_path: str):
        print(f"  [Dataset] {h5_path}")
        with open(map_path, 'rb') as f:
            self.map_data = pickle.load(f)['Map']

        with h5py.File(h5_path, 'r') as f:
            self.hist        = torch.from_numpy(f['HISTORY'][:]).float()
            self.fut         = torch.from_numpy(f['FUTURE'][:]).float()
            self.pos         = torch.from_numpy(f['POS'][:]).float()
            self.valid_len   = torch.from_numpy(f['VALID_LEN'][:]).long()
            self.norm_center = torch.from_numpy(f['NORM_CENTER'][:]).float()
            self.theta       = torch.from_numpy(f['THETA'][:]).float()
            lane_ids         = f['LANE_ID'][:]
            city_names       = [c.decode('utf-8') if isinstance(c, bytes) else str(c)
                                for c in f['CITY_NAME'][:]]

        N, max_lanes = lane_ids.shape
        lane_tensor = np.zeros((N, max_lanes, 10, 5), dtype=np.float32)
        for i in tqdm(range(N), desc="  Assembling lanes", leave=False):
            city_map = self.map_data[city_names[i]]
            for j, lid in enumerate(lane_ids[i]):
                if lid != -1:
                    lane_tensor[i, j] = city_map[lid]
        self.lanes = torch.from_numpy(lane_tensor).float()
        print(f"  → {N:,} samples")

    def __len__(self):
        return len(self.hist)

    def __getitem__(self, idx):
        return {
            'HISTORY':     self.hist[idx],
            'FUTURE':      self.fut[idx],
            'POS':         self.pos[idx],
            'LANE':        self.lanes[idx],
            'VALID_LEN':   self.valid_len[idx],
            'NORM_CENTER': self.norm_center[idx],
            'THETA':       self.theta[idx],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Loss & Metrics
# ──────────────────────────────────────────────────────────────────────────────
def compute_loss(pred_trajs, pred_confs, gt_trajs):
    B = pred_trajs.shape[0]
    dist_ep    = torch.norm(pred_trajs[:, :, -1, :] - gt_trajs[:, -1, :].unsqueeze(1), dim=-1)
    best_k_idx = torch.argmin(dist_ep, dim=-1)
    best_pred  = pred_trajs[torch.arange(B), best_k_idx]
    loss_reg   = F.smooth_l1_loss(best_pred, gt_trajs)

    if pred_confs.min() >= 0 and pred_confs.max() <= 1:
        loss_cls = F.nll_loss(torch.log(pred_confs + 1e-9), best_k_idx)
    else:
        loss_cls = F.cross_entropy(pred_confs, best_k_idx)

    return loss_reg + loss_cls, loss_reg, loss_cls, best_k_idx


def compute_metrics(pred_traj, gt_traj):
    ade  = torch.norm(pred_traj - gt_traj, dim=-1).mean().item()
    rmse = torch.sqrt(
        torch.pow(pred_traj - gt_traj, 2).sum(-1).mean(-1)
    ).mean().item()
    return ade, rmse


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
def build_model(cfg: dict, device: torch.device) -> mmTrans:
    feature_mode = cfg['exp']['feature_mode']
    in_channels  = 4 + len(EXTRA_FEATURE_MAP[feature_mode])

    model_cfg = dict(cfg['model'])
    model_cfg['in_channels'] = in_channels
    return mmTrans(STF, model_cfg).to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
def train(cfg: dict):
    # ── 기본 설정 읽기 ────────────────────────────────────────────────────────
    exp_cfg   = cfg['exp']
    data_cfg  = cfg['data']
    train_cfg = cfg['train']

    cond         = exp_cfg['cond']
    feature_mode = exp_cfg['feature_mode']
    seed         = train_cfg['seed']

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    # ── 경로 ─────────────────────────────────────────────────────────────────
    train_h5  = data_cfg['train']['processed_data_path']
    train_map = data_cfg['train']['processed_maps_path']
    val_h5    = data_cfg['val']['processed_data_path']
    val_map   = data_cfg['val']['processed_maps_path']

    ckpt_dir  = Path(train_cfg['ckpt_dir']) / cfg['_config_stem']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    log_dir = Path("logs") / cfg['_config_stem'] / datetime.now().strftime("%m%d-%H%M")
    writer  = SummaryWriter(log_dir=str(log_dir))

    print(f"\n{'='*60}")
    print(f"  cond={cond}  feature_mode={feature_mode}  seed={seed}")
    print(f"  in_channels={4 + len(EXTRA_FEATURE_MAP[feature_mode])}")
    print(f"  ckpt → {best_path}")
    print(f"  log  → {log_dir}")
    print(f"{'='*60}")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    train_ds = HighDDataset(train_h5, train_map)
    val_ds   = HighDDataset(val_h5,   val_map)

    batch_size         = data_cfg.get('batch_size', 512)
    num_workers        = data_cfg.get('num_workers', 8)
    persistent_workers = train_cfg.get('persistent_workers', True)
    num_epochs         = train_cfg.get('epochs', 200)
    use_amp            = train_cfg.get('use_amp', True)
    lr                 = float(train_cfg['lr'])

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

    # ── Model / Optimizer ────────────────────────────────────────────────────
    raw_model = build_model(cfg, device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr,
                                  weight_decay=float(train_cfg.get('weight_decay', 0)))
    scaler    = GradScaler('cuda') if use_amp else None
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=len(train_loader) * num_epochs,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10, final_div_factor=100,
    )
    model     = torch.compile(raw_model)
    best_rmse = float('inf')

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        model.train()
        t_loss = t_ade = t_rmse = 0.0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{num_epochs} Train", leave=False)

        for bi, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = autocast(device_type='cuda') if use_amp else nullcontext()
            with ctx:
                pred, conf = model(batch)
                target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
                target_conf = conf[:, 0, ...] if conf.dim() == 3 else conf
                target_gt   = batch['FUTURE'][:, 0, :, :2]
                loss, _, _, bki = compute_loss(target_pred, target_conf, target_gt)
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

            gs = (epoch - 1) * len(train_loader) + bi
            writer.add_scalar('Train/Loss', loss.item(), gs)
            writer.add_scalar('Train/LR',   optimizer.param_groups[0]['lr'], gs)

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

        writer.add_scalar('Loss/Train', t_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Val',   avg_v_loss, epoch)
        writer.add_scalar('Metric/ADE', avg_v_ade,  epoch)
        writer.add_scalar('Metric/RMSE',avg_v_rmse, epoch)

        print(f"  E{epoch:3d} | val_loss={avg_v_loss:.4f}  "
              f"ADE={avg_v_ade:.4f}  RMSE={avg_v_rmse:.4f}")

        if avg_v_rmse < best_rmse:
            best_rmse = avg_v_rmse
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    raw_model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'val_rmse':            best_rmse,
                'val_ade':             avg_v_ade,
                'cond':                cond,
                'feature_mode':        feature_mode,
                'seed':                seed,
            }, best_path)
            print(f"  ⭐ Best → RMSE={best_rmse:.4f}  [{best_path}]")

    writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='실험 config yaml 경로 (예: configs/c0_seed42.yaml)')
    args = parser.parse_args()

    cfg = load_config_data(args.config)
    cfg['_config_stem'] = Path(args.config).stem   # e.g. "c0_seed42"
    train(cfg)