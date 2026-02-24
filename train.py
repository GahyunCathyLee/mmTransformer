import os
import pickle
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from lib.utils.utilities import load_config_data, save_checkpoint, load_model_class
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

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
    
    # 💡 최적화 1: GPU 연산 알고리즘 최적화
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True 
        
    cfg = load_config_data(args.config)
    stacked_transformer_class = STF
    
    # 설정값 병합 및 모델 파라미터 강제 고정
    model_cfg = {}
    for key in ['data', 'model', 'train']:
        if key in cfg and isinstance(cfg[key], dict):
            model_cfg.update(cfg[key])
    for k, v in cfg.items():
        if not isinstance(v, dict):
            model_cfg[k] = v
    
    # 💡 텐서 사이즈 충돌 방지 고정
    model_cfg['max_lane_num'] = 6
    model_cfg['max_agent_num'] = 9
    model_cfg['lane_channels'] = 7
            
    model = mmTrans(stacked_transformer_class, model_cfg).to(device)

    if args.resume:
        config_name = os.path.basename(args.config).replace('.yaml', '')
        ckpt_path = os.path.join(cfg.get('train', {}).get('ckpt_dir', './ckpts/baseline'), config_name, 'best.pt')
        
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            else:
                model.load_state_dict(checkpoint)

            print(f"🔄 [Resume] 가중치 복구가 완료되었습니다.")
        else:
            print(f"⚠️ [Warning] {ckpt_path}에 체크포인트가 없어 처음부터 학습을 시작합니다.")

    # 💡 모델 내부 속성 강제 주입
    model.max_lane_num = 6
    model.max_agent_num = 9
    
    print_model_size(model)
    

    base_lr = float(cfg.get('train', {}).get('lr', 1e-4))
    lr = base_lr * 0.2 if args.resume else base_lr
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.resume:
        print(f"📉 [LR Adjust] 학습을 재개하므로 학습률을 {base_lr} -> {lr}로 낮췄습니다.")
    
    # 💡 최적화 2: AMP Scaler 초기화 (최신 문법)
    scaler = GradScaler('cuda')
    
    # 데이터셋 로드 (In-Memory 방식 권장)
    base_dir = args.data_dir
    train_dataset = HighDDataset(data_path=f'{base_dir}/train.h5', map_path=f'{base_dir}/map.pkl')
    val_dataset = HighDDataset(data_path=f'{base_dir}/val.h5', map_path=f'{base_dir}/map.pkl')
    
    batch_size = cfg.get('data', {}).get('batch_size', 512)
    num_workers = 32 
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )

    num_epochs = cfg.get('train', {}).get('epochs', 500)
    
    best_rmse = float('inf') 
    config_name = os.path.basename(args.config).replace('.yaml', '')
    save_dir = os.path.join(cfg.get('train', {}).get('ckpt_dir', './checkpoints'), config_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Start Training on {device} with AMP Enabled...\n")
    
    for epoch in range(num_epochs):
        # --- [TRAIN PHASE] ---
        model.train()
        train_loss, train_ade, train_rmse = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_data in pbar:
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
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
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            scaler.step(optimizer)
            scaler.update()

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

                with autocast(device_type='cuda'):
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
        
        print(f"Epoch [{epoch+1}] Val Loss: {avg_v_loss:.4f} | ADE: {avg_v_ade:.4f} | RMSE: {avg_v_rmse:.4f}")

        if avg_v_rmse < best_rmse:
            best_rmse = avg_v_rmse
            save_path = os.path.join(save_dir, 'best.pt')
            
            save_checkpoint(save_path, model, optimizer, MR=best_rmse)
            print(f"⭐ Best RMSE Updated! ({best_rmse:.4f}) Model saved to {save_path}")

        print("-" * 30)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml', help='path to config file')
    parser.add_argument('--data_dir', type=str, default='highD/baseline', help='path to data folder containing h5 and map.pkl')
    parser.add_argument('--resume', action='store_true', help='resume from best checkpoint')
    args = parser.parse_args()
    
    train(args)