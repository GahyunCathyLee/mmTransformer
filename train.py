import os
import pickle
import numpy as np
import h5py
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter # 💡 추가
from datetime import datetime
from contextlib import nullcontext

from lib.utils.utilities import load_config_data
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

# [0:dx, 1:dy, 2:dvx, 3:dvy, 4:ax, 5:ay, 6:lc_state, 7:dx_time, 8:gate]
EXTRA_FEATURE_MAP = {
    'baseline': [0, 1],              
    'exp1': [0, 1, 6, 7],                  
    'exp2': [0, 1, 4, 5, 6, 7, 8],               
    'exp3': [0, 1, 2, 3, 4, 5, 8],               
    'exp4': [0, 1, 2, 3, 4, 5, 6, 7, 8],   
    'exp5': [0, 1, 6, 7, 8],
    'exp6': [6, 7],
    'exp7': [4, 5, 6, 7, 8],
    'exp8': [0, 1, 6, 8],
    'exp9': [0, 1, 8]            
}


# ==============================================================================
# 1. Dataset 구성 (In-Memory 캐싱 적용)
# ==============================================================================

class HighDDataset(Dataset):
    def __init__(self, data_path, map_path):
        self.h5_path = data_path
        
        with open(map_path, 'rb') as f:
            map_info = pickle.load(f)
            self.map_data = map_info['Map']
        
        print(f"[{data_path}] RAM에 데이터를 올리는 중...")
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f['HISTORY'])
            
            self.hist = torch.from_numpy(f['HISTORY'][:]).float()
            self.fut = torch.from_numpy(f['FUTURE'][:]).float()
            self.pos = torch.from_numpy(f['POS'][:]).float()
            self.valid_len = torch.from_numpy(f['VALID_LEN'][:]).long()
            self.norm_center = torch.from_numpy(f['NORM_CENTER'][:]).float()
            self.theta = torch.from_numpy(f['THETA'][:]).float()
            
            lane_ids = f['LANE_ID'][:]
            city_names_raw = f['CITY_NAME'][:]
            city_names = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in city_names_raw]
            
        print(f"[{data_path}] 차선(Lane) 피처 사전 병합 중...")
        max_lanes = lane_ids.shape[1]
        lane_tensor_np = np.zeros((self.length, max_lanes, 10, 5), dtype=np.float32)
        
        # tqdm으로 진행률 표시
        for i in tqdm(range(self.length), desc="Assembling Lanes"):
            city_map = self.map_data[city_names[i]]
            for j, l_id in enumerate(lane_ids[i]):
                if l_id != -1:
                    lane_tensor_np[i, j] = city_map[l_id]
                    
        self.lanes = torch.from_numpy(lane_tensor_np).float()
        print(f"[{data_path}] 모든 데이터 PyTorch Tensor 적재 완료!\n")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            'HISTORY': self.hist[idx],
            'FUTURE': self.fut[idx],
            'POS': self.pos[idx],
            'LANE': self.lanes[idx],
            'VALID_LEN': self.valid_len[idx],
            'NORM_CENTER': self.norm_center[idx],
            'THETA': self.theta[idx]
        }

# ==============================================================================
# 2. Loss Function 정의 (Best-of-K)
# ==============================================================================
def compute_loss(pred_trajs, pred_confs, gt_trajs):
    """
    pred_trajs: [B, K, T, 2]
    pred_confs: [B, K] 
    gt_trajs: [B, T, 2]
    """
    B, K, T, _ = pred_trajs.shape

    # 1. Best-of-K (마지막 프레임 FDE 기준)
    pred_endpoints = pred_trajs[:, :, -1, :]  # [B, K, 2]
    gt_endpoints = gt_trajs[:, -1, :].unsqueeze(1)  # [B, 1, 2]
    
    distances = torch.norm(pred_endpoints - gt_endpoints, dim=-1)  
    best_k_idx = torch.argmin(distances, dim=-1)  

    # 2. Regression Loss (Smooth L1)
    best_pred_trajs = pred_trajs[torch.arange(B), best_k_idx]  
    loss_reg = F.smooth_l1_loss(best_pred_trajs, gt_trajs)

    # 3. Classification Loss (NLL Loss)
    loss_cls = F.nll_loss(pred_confs, best_k_idx)

    total_loss = loss_reg + loss_cls
    return total_loss, loss_reg, loss_cls

def compute_metrics(pred_traj, gt_traj):
    """
    pred_traj: [B, T, 2] (Best-of-K로 선택된 단일 궤적)
    gt_traj: [B, T, 2]
    """
    # 1. ADE: 모든 타임스텝의 평균 유클리드 거리
    dist = torch.norm(pred_traj - gt_traj, dim=-1) # [B, T]
    ade = dist.mean() 
    
    # 2. RMSE: 각 배치별 평균 제곱 오차의 루트
    mse = torch.pow(pred_traj - gt_traj, 2).sum(dim=-1).mean(dim=-1) # [B]
    rmse = torch.sqrt(mse).mean()
    
    return ade.item(), rmse.item()

# ==============================================================================
# 3. Training Loop (초고속 최적화 버전)
# ==============================================================================
def print_model_size(model):
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"\n" + "="*50)
    print(f"📊 Model Size Info")
    print(f"  • Total Parameters : {param_count:,}")
    print(f"  • Model Memory Size: {size_all_mb:.2f} MB")
    print("="*50 + "\n")

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config_data(args.config)
    
    # ✅ 2. 실험 모드 및 채널 수 자동 계산
    # Config에서 feature_mode를 가져오고, 없으면 baseline 사용
    feature_mode = cfg.get('exp', {}).get('feature_mode', 'baseline')
    num_extra = len(EXTRA_FEATURE_MAP[feature_mode])
    in_channels = 4 + num_extra  # 기본(x,y,t,m) + 추가 피처 개수
    
    # ✅ 3. 경로 설정 자동화
    data_dir = Path(args.data_dir) / feature_mode
    save_dir = Path(cfg.get('train', {}).get('ckpt_dir', './checkpoints')) / feature_mode
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"

    log_time = datetime.now().strftime("%m%d-%H%M")
    log_dir = Path("logs") / feature_mode / log_time
    writer = SummaryWriter(log_dir=str(log_dir))
    

    print("=" * 50)
    print(f"🚀 Experiment Mode : {feature_mode}")
    print(f"📏 Input Channels  : {in_channels} (4 + {num_extra})")
    print(f"📂 Data Directory  : {data_dir}")
    print(f"💾 Save Directory  : {save_dir}")
    print(f"📊 TensorBoard Log : {log_dir}")
    print("=" * 50)

    # 모델 설정 업데이트
    model_cfg = cfg.get('model', {})
    model_cfg['in_channels'] = in_channels
    model_cfg['max_lane_num'] = 6
    model_cfg['max_agent_num'] = 9
    
    future_frames = model_cfg.get('future_num_frames', 25)
    model_cfg['out_channels'] = future_frames * 2
    
    model = mmTrans(STF, model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get('train', {}).get('lr', 1e-4)))
    
    use_amp = cfg.get('train', {}).get('use_amp', True)
    scaler = GradScaler('cuda') if use_amp else None

    train_dataset = HighDDataset(data_path=str(data_dir / 'train.h5'), map_path=str(data_dir / 'map.pkl'))
    val_dataset = HighDDataset(data_path=str(data_dir / 'val.h5'), map_path=str(data_dir / 'map.pkl'))
    
    batch_size = cfg.get('data', {}).get('batch_size', 1024)
    num_workers = cfg.get('data', {}).get('num_workers', 32)
    num_epochs = cfg.get('train', {}).get('epochs', 500)
    persistent_workers = cfg.get('train', {}).get('persistent_workers', True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers
    )

    total_steps = len(train_loader) * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(cfg.get('train', {}).get('lr')),
        total_steps=total_steps,
        pct_start=0.1,    
        anneal_strategy='cos',
        div_factor=10,    
        final_div_factor=100 
    )

    start_epoch = 1
    best_rmse = float('inf')

    # ✅ 4. 자동 Resume 로직 
    if args.resume:
        if best_path.exists():
            print(f"🔄 [Resume] 기존 체크포인트 로드: {best_path}")
            checkpoint = torch.load(best_path, map_location=device)
            
            # 가중치 복구 (다양한 키값 대응)
            state_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'state_dict'
            model.load_state_dict(checkpoint[state_key])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_rmse = checkpoint.get('val_rmse', float('inf'))
            print(f"✅ {checkpoint.get('epoch')} 에포크부터 재개합니다. (Best RMSE: {best_rmse:.4f})")
        else:
            print(f"⚠️ [Resume] 체크포인트를 찾을 수 없어 처음부터 학습을 시작합니다.")

    print(f"Start Training on {device} with AMP Enabled...\n")
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"{'=' * 30}  Epoch {epoch}/{num_epochs}  {'=' * 30}")
        # --- [TRAIN PHASE] ---
        model.train()
        train_loss, train_ade, train_rmse = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"[Train]")
        
        for batch_idx, batch_data in enumerate(pbar):
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            context = autocast(device_type='cuda') if use_amp else nullcontext()

            with context:
                pred, conf = model(batch_data)
                
                # Target Agent 추출
                target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
                target_conf = conf[:, 0, ...] if conf.dim() == 3 else conf
                target_gt = batch_data['FUTURE'][:, 0, :, :2]

                # Best-of-K 선택을 위한 Loss 및 Index 계산
                loss, l_reg, l_cls = compute_loss(target_pred, target_conf, target_gt)

                if torch.isnan(loss):
                    print("⚠️ NaN Loss detected! Skipping this batch...")
                    optimizer.zero_grad(set_to_none=True)
                    continue
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
            else:
                loss.backward() # 일반적인 역전파
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            curr_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', loss.item(), curr_step)
            writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], curr_step)

            # Metric 계산 (Best-of-K 기반)
            with torch.no_grad():
                dist_last = torch.norm(target_pred[:, :, -1, :] - target_gt[:, -1, :].unsqueeze(1), dim=-1)
                best_idx = torch.argmin(dist_last, dim=-1)
                best_traj = target_pred[torch.arange(target_pred.size(0)), best_idx]
                
                ade, rmse = compute_metrics(best_traj, target_gt)
                train_loss += loss.item()
                train_ade += ade
                train_rmse += rmse

            pbar.set_postfix({'A': f"{ade:.3f}", 'R': f"{rmse:.3f}"})

        # --- [VAL PHASE] ---
        model.eval()
        val_loss, val_ade, val_rmse = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                for k, v in batch_data.items():
                    if isinstance(v, torch.Tensor):
                        batch_data[k] = v.to(device, non_blocking=True)

                context = autocast(device_type='cuda') if use_amp else nullcontext()
                with context:
                    pred, conf = model(batch_data)
                    target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
                    target_conf = conf[:, 0, ...] if conf.dim() == 3 else conf
                    target_gt = batch_data['FUTURE'][:, 0, :, :2]

                    loss, _, _ = compute_loss(target_pred, target_conf, target_gt)
                
                # Best 궤적 선택 후 Metric 계산
                dist_last_v = torch.norm(target_pred[:, :, -1, :] - target_gt[:, -1, :].unsqueeze(1), dim=-1)
                best_idx_v = torch.argmin(dist_last_v, dim=-1)
                best_traj_v = target_pred[torch.arange(target_pred.size(0)), best_idx_v]
                
                v_ade, v_rmse = compute_metrics(best_traj_v, target_gt)
                
                val_loss += loss.item()
                val_ade += v_ade
                val_rmse += v_rmse
                
        # 평균 지표 계산
        avg_v_loss = val_loss / len(val_loader)
        avg_v_ade = val_ade / len(val_loader)
        avg_v_rmse = val_rmse / len(val_loader)

        writer.add_scalar('Loss/Train_Epoch', train_loss / len(train_loader), epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_v_loss, epoch)
        writer.add_scalar('Metric/Val_ADE', avg_v_ade, epoch)
        writer.add_scalar('Metric/Val_RMSE', avg_v_rmse, epoch)
        
        print(f"Epoch [{epoch+1}] Val Loss: {avg_v_loss:.4f} | ADE: {avg_v_ade:.4f} | RMSE: {avg_v_rmse:.4f}")

        if avg_v_rmse < best_rmse:
            best_rmse = avg_v_rmse
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_rmse,
                'feature_mode': feature_mode
            }, best_path)
            print(f"⭐ Best RMSE Updated! ({best_rmse:.4f}) -> {best_path}")
        print()

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='path to config file')
    parser.add_argument('--data_dir', type=str, default='highD', help='path to data folder containing h5 and map.pkl')
    parser.add_argument('--resume', action='store_true', help='resume from best checkpoint')
    args = parser.parse_args()
    
    train(args)
    