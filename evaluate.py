import os
import csv
import pickle
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torch.amp import autocast
import time
from pathlib import Path

from lib.utils.utilities import load_config_data
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

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
# 1. Dataset (train.py와 동일 구조)
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
# 2. Metric 계산 함수들
# ==============================================================================
def compute_metrics_detailed(pred_trajs, gt_trajs, fps=5):
    """
    pred_trajs : [B, K, T, 2]
    gt_trajs   : [B, T, 2]
    fps        : sampling rate (현재 5Hz)

    Returns:
        minADE, minFDE,
        RMSE (전체 평균),
        RMSE@1s ... RMSE@5s (정확한 시점 기반)
    """

    B, K, T, _ = pred_trajs.shape

    # ─────────────────────────────────────────────
    # Best-of-K selection (minFDE 기준)
    # ─────────────────────────────────────────────
    pred_endpoints = pred_trajs[:, :, -1, :]
    gt_endpoints   = gt_trajs[:, -1, :].unsqueeze(1)
    dist_endpoint  = torch.norm(pred_endpoints - gt_endpoints, dim=-1)
    best_idx       = torch.argmin(dist_endpoint, dim=-1)
    best_traj      = pred_trajs[torch.arange(B), best_idx]  # [B, T, 2]

    # ─────────────────────────────────────────────
    # minADE / minFDE
    # ─────────────────────────────────────────────
    dist_all  = torch.norm(pred_trajs - gt_trajs.unsqueeze(1), dim=-1)
    ade_per_k = dist_all.mean(dim=-1)
    min_ade   = ade_per_k.min(dim=-1).values
    min_fde   = dist_endpoint.min(dim=-1).values

    # ─────────────────────────────────────────────
    # RMSE (전체 구간 평균)
    # ─────────────────────────────────────────────
    sq_err = torch.pow(best_traj - gt_trajs, 2).sum(dim=-1)  # [B, T]
    rmse_overall = torch.sqrt(sq_err.mean(dim=-1))  # [B]

    # ─────────────────────────────────────────────
    # 논문 스타일 RMSE@Ns (정확한 시점 기반)
    # ─────────────────────────────────────────────
    rmse_at = {}

    for s in range(1, 6):  # 1초 ~ 5초
        step_idx = int(s * fps) - 1  # index 보정
        step_idx = min(step_idx, T - 1)

        rmse_s = torch.sqrt(sq_err[:, step_idx])
        rmse_at[f'RMSE@{s}s'] = rmse_s

    return {
        'minADE'     : min_ade,
        'minFDE'     : min_fde,
        'RMSE'       : rmse_overall,
        **rmse_at,
    }


# ==============================================================================
# 3. 체크포인트 로드 헬퍼
# ==============================================================================
def load_checkpoint(ckpt_path, model, device):
    print(f"📂 체크포인트 로드: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # save_checkpoint 포맷에 따라 키가 다를 수 있으므로 유연하게 처리
    if 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt  # 통째로 state_dict인 경우

    model.load_state_dict(state, strict=True)
    saved_metric = ckpt.get('MR', ckpt.get('rmse', None))
    if saved_metric is not None:
        print(f"   └─ 저장된 Best RMSE: {saved_metric:.4f}")
    return model


# ==============================================================================
# 4. CSV 저장 헬퍼
# ==============================================================================
def save_csv(result_dict, save_path):
    fieldnames = list(result_dict.keys())
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(result_dict)
    print(f"📄 결과 CSV 저장 완료: {save_path}")


# ==============================================================================
# 3. Inference Time Measurement Function
# ==============================================================================
def measure_inference_time(model, dataset, device, num_iters=10000):
    """
    Batch Size 1로 num_iters만큼 추론 시간을 측정합니다.
    """
    model.eval()
    # Batch size 1로 하나의 샘플 추출
    sample = dataset[0]
    batch_data = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch_data[k] = v.unsqueeze(0).to(device)

    latencies = []
    print(f"\n⏱️ Inference Time 측정 시작 (Iterations: {num_iters}, Batch Size: 1)")
    
    with torch.no_grad():
        # Warm-up (100 iters)
        for _ in range(100):
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                _ = model(batch_data)
        
        # 실제 측정
        for _ in tqdm(range(num_iters), desc="Measuring Latency"):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                _ = model(batch_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # ms 단위 저장

    latencies = np.array(latencies)
    avg_t = np.mean(latencies)
    min_t = np.min(latencies)
    max_t = np.max(latencies)
    std_t = np.std(latencies)

    print(f"\n" + "="*50)
    print(f" 🏁 Inference Time 결과 (Batch Size: 1)")
    print(f"  - Average : {avg_t:.4f} ms")
    print(f"  - Minimum : {min_t:.4f} ms")
    print(f"  - Maximum : {max_t:.4f} ms")
    print(f"  - Std Dev : {std_t:.4f} ms")
    print("="*50 + "\n")
    
    return {'avg': avg_t, 'min': min_t, 'max': max_t}

# ==============================================================================
# 5. 평가 메인 함수
# ==============================================================================
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config_data(args.config)

    # ✅ 실험 모드 및 채널 자동 설정 (train.py와 동기화)
    feature_mode = cfg.get('exp', {}).get('feature_mode', 'baseline')
    num_extra = len(EXTRA_FEATURE_MAP[feature_mode])
    in_channels = 4 + num_extra
    
    model_cfg = cfg.get('model', {})
    model_cfg['in_channels'] = in_channels
    model_cfg['max_lane_num'], model_cfg['max_agent_num'] = 6, 9
    model_cfg['lane_channels'] = 7
    model_cfg['out_channels'] = model_cfg.get('future_num_frames', 25) * 2

    model = mmTrans(STF, model_cfg).to(device)

    # ✅ 체크포인트 경로 자동 탐색
    if args.ckpt:
        ckpt_path = args.ckpt
    else:
        ckpt_dir = Path(cfg.get('train', {}).get('ckpt_dir', './ckpts'))
        ckpt_path = ckpt_dir / feature_mode / "best.pt"

    print(f"📂 평가 모델: {ckpt_path} (Mode: {feature_mode})")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint)))
    model.eval()

    # ✅ 데이터 경로 자동 설정
    data_dir = Path(args.data_dir) / feature_mode
    dataset = HighDDataset(str(data_dir / f"{args.split}.h5"), str(data_dir / "map.pkl"))
    loader = DataLoader(dataset, batch_size=cfg.get('data', {}).get('batch_size', 512), 
                        shuffle=False, num_workers=8, pin_memory=True)

    # --- Mode 1: Inference Time Measurement ---
    if args.measure_time:
        measure_inference_time(model, dataset, device, num_iters=10000)
        return

    metric_keys = ['minADE', 'minFDE', 'RMSE'] + [f'RMSE@{s}s' for s in range(1, 6)]
    accum = {k: 0.0 for k in metric_keys}
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Eval [{args.split}]")
        for batch_data in pbar:
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor): batch_data[k] = v.to(device)

            with autocast(device_type='cuda'):
                pred, _ = model(batch_data)
            
            target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred
            target_gt = batch_data['FUTURE'][:, 0, :, :2]
            
            metrics = compute_metrics_detailed(target_pred, target_gt)
            for k in metric_keys: accum[k] += metrics[k].sum().item()
            total_samples += target_gt.size(0)

    final = {k: accum[k] / total_samples for k in metric_keys}
    
    header = f" 🏁 Final Evaluation Result: [{feature_mode}] "
    print("\n" + " " * 10 + "●" * 40)
    print(f"{header:^60}")
    print(" " * 10 + "●" * 40 + "\n")

    # 1. 핵심 지표 (Overall Metrics)
    print(f"  📂 Target Split  : {args.split}")
    print(f"  🔢 Total Samples : {total_samples:,}")
    print(f"  📍 Checkpoint    : {Path(ckpt_path).name}")
    print("-" * 50)
    
    print(f"  🔥 minADE        : {final['minADE']:.4f} m")
    print(f"  🔥 minFDE        : {final['minFDE']:.4f} m")
    print(f"  🔥 RMSE (Total)  : {final['RMSE']:.4f} m")
    
    print("-" * 50)
    
    # 2. 시간대별 상세 지표 (Time-step Metrics)
    # 논문에 들어갈 RMSE@Ns 수치를 테이블 형태로 출력합니다.
    print(f"  🕒 Time-step Analysis (RMSE)")
    print(f"  {'-' * 32}")
    print(f"  |  1.0s  |  2.0s  |  3.0s  |  4.0s  |  5.0s  |")
    print(f"  | {final['RMSE@1s']:^6.3f} | {final['RMSE@2s']:^6.3f} | {final['RMSE@3s']:^6.3f} | {final['RMSE@4s']:^6.3f} | {final['RMSE@5s']:^6.3f} |")
    print(f"  {'-' * 32}")

    print("\n" + "=" * 50 + "\n")

    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, f"eval_{feature_mode}_{args.split}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mode'] + metric_keys)
            writer.writeheader()
            writer.writerow({'mode': feature_mode, **final})
        print(f"📄 CSV Saved: {csv_path}")

    return final


# ==============================================================================
# 6. Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    parser.add_argument('--data_dir', type=str, default='highD')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--measure_time', action='store_true')
    parser.add_argument('--save_csv', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, default='./results')
    
    evaluate(args := parser.parse_args())