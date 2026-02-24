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

from lib.utils.utilities import load_config_data, load_model_class
from lib.models.mmTransformer import mmTrans
from lib.models.TF_version.stacked_transformer import STF

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

            self.hist       = torch.from_numpy(f['HISTORY'][:]).float()
            self.fut        = torch.from_numpy(f['FUTURE'][:]).float()
            self.pos        = torch.from_numpy(f['POS'][:]).float()
            self.valid_len  = torch.from_numpy(f['VALID_LEN'][:]).long()
            self.norm_center= torch.from_numpy(f['NORM_CENTER'][:]).float()
            self.theta      = torch.from_numpy(f['THETA'][:]).float()

            lane_ids       = f['LANE_ID'][:]
            city_names_raw = f['CITY_NAME'][:]
            city_names     = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in city_names_raw]

        print(f"[{data_path}] 차선(Lane) 피처 사전 병합 중...")
        max_lanes = lane_ids.shape[1]
        lane_tensor_np = np.zeros((self.length, max_lanes, 10, 5), dtype=np.float32)

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
            'HISTORY'    : self.hist[idx],
            'FUTURE'     : self.fut[idx],
            'POS'        : self.pos[idx],
            'LANE'       : self.lanes[idx],
            'VALID_LEN'  : self.valid_len[idx],
            'NORM_CENTER': self.norm_center[idx],
            'THETA'      : self.theta[idx],
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
# 5. 평가 메인 함수
# ==============================================================================
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    cfg = load_config_data(args.config)

    # ── 모델 설정 병합 (train.py와 동일) ──────────────────────────────────────
    model_cfg = {}
    for key in ['data', 'model', 'train']:
        if key in cfg and isinstance(cfg[key], dict):
            model_cfg.update(cfg[key])
    for k, v in cfg.items():
        if not isinstance(v, dict):
            model_cfg[k] = v

    model_cfg['max_lane_num']   = 6
    model_cfg['max_agent_num']  = 9
    model_cfg['lane_channels']  = 7

    # ── 모델 초기화 ───────────────────────────────────────────────────────────
    stacked_transformer_class = STF
    model = mmTrans(stacked_transformer_class, model_cfg).to(device)
    model.max_lane_num  = 6
    model.max_agent_num = 9

    # ── 체크포인트 경로 결정 ──────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        config_name = os.path.basename(args.config).replace('.yaml', '')
        ckpt_dir    = cfg.get('train', {}).get('ckpt_dir', './checkpoints')
        ckpt_path   = os.path.join(ckpt_dir, config_name, 'best.pt')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

    model = load_checkpoint(ckpt_path, model, device)
    model.eval()

    # ── 데이터셋 로드 ─────────────────────────────────────────────────────────
    split      = args.split  # 'test' or 'val'
    data_file  = os.path.join(args.data_dir, f'{split}.h5')
    map_file   = os.path.join(args.data_dir, 'map.pkl')
    dataset    = HighDDataset(data_path=data_file, map_path=map_file)

    batch_size  = cfg.get('data', {}).get('batch_size', 512)
    num_workers = min(32, os.cpu_count() or 4)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=4, persistent_workers=True,
    )

    # ── 평가 루프 ─────────────────────────────────────────────────────────────
    metric_keys = ['minADE', 'minFDE', 'RMSE',
                   'RMSE@1s', 'RMSE@2s', 'RMSE@3s', 'RMSE@4s', 'RMSE@5s']
    accum = {k: 0.0 for k in metric_keys}
    total_samples = 0

    print(f"\n🚀 평가 시작 — split: [{split}] | device: {device}\n")

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Evaluating [{split}]")
        for batch_data in pbar:
            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.to(device, non_blocking=True)

            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                pred, conf = model(batch_data)

            # Target agent 추출 (train.py와 동일한 로직)
            target_pred = pred[:, 0, ...] if pred.dim() == 5 else pred   # [B, K, T, 2]
            target_gt   = batch_data['FUTURE'][:, 0, :, :2]              # [B, T, 2]

            B = target_gt.size(0)
            metrics = compute_metrics_detailed(target_pred, target_gt)

            for k in metric_keys:
                accum[k] += metrics[k].sum().item()
            total_samples += B

            # tqdm 실시간 표시 (배치 평균)
            pbar.set_postfix({
                'minADE': f"{metrics['minADE'].mean().item():.3f}",
                'minFDE': f"{metrics['minFDE'].mean().item():.3f}",
                'RMSE'  : f"{metrics['RMSE'].mean().item():.3f}",
            })

    # ── 최종 평균 집계 ────────────────────────────────────────────────────────
    final = {k: accum[k] / total_samples for k in metric_keys}

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  평가 결과 | split: {split} | samples: {total_samples:,}")
    print(sep)
    print(f"  {'minADE':<14}: {final['minADE']:.4f} m")
    print(f"  {'minFDE':<14}: {final['minFDE']:.4f} m")
    print(f"  {'RMSE':<14}: {final['RMSE']:.4f} m")
    print(f"  {'-'*44}")
    for s in range(1, 6):
        key = f'RMSE@{s}s'
        print(f"  {key:<14}: {final[key]:.4f} m")
    print(f"{sep}\n")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_tag  = os.path.splitext(os.path.basename(ckpt_path))[0]
        csv_name  = f"eval_{split}_{ckpt_tag}.csv"
        csv_path  = os.path.join(args.output_dir, csv_name)

        csv_row = {'split': split, 'checkpoint': ckpt_path,
                   'num_samples': total_samples, **final}
        save_csv(csv_row, csv_path)

    return final


# ==============================================================================
# 6. Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mmTransformer Evaluation on highD")

    parser.add_argument('--config',
                        type=str, default='configs/baseline.yaml',
                        help='학습에 사용한 config 파일 경로')
    parser.add_argument('--data_dir',
                        type=str, default='highD/baseline',
                        help='h5 및 map.pkl이 위치한 데이터 폴더')
    parser.add_argument('--split',
                        type=str, default='test', choices=['val', 'test'],
                        help='평가할 데이터 split (val 또는 test)')

    # 체크포인트: 미지정 시 save_dir/config_name/best.pt 자동 탐색
    parser.add_argument('--checkpoint',
                        type=str, default="ckpts/baseline/best.pt",
                        help='체크포인트 경로 (.pt). 미지정 시 best.pt 자동 로드')

    # 결과 저장
    parser.add_argument('--save_csv',
                        action='store_true', default=True,
                        help='결과를 CSV로 저장 (기본값: True)')
    parser.add_argument('--output_dir',
                        type=str, default='./results',
                        help='CSV 저장 디렉토리')

    args = parser.parse_args()
    evaluate(args)