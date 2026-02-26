import os
import re
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import h5py
from concurrent.futures import ProcessPoolExecutor
import traceback

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
TARGET_FPS = 5.0
T_H = 15
T_F = 25
SEQ_LEN = T_H + T_F
MAX_AGENTS = 9  
MAX_LANES = 6   
LANE_PTS = 10    

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="highD/raw")
    parser.add_argument("--out_dir", type=str, default="highD") 
    parser.add_argument("--feature_mode", type=str, choices=EXTRA_FEATURE_MAP.keys(), default='baseline')
    parser.add_argument("--t_front", type=float, default=3.0)
    parser.add_argument("--t_back", type=float, default=5.0)
    parser.add_argument("--vy_eps", type=float, default=0.27)
    parser.add_argument("--eps_gate", type=float, default=0.1)
    parser.add_argument("--slide_window_sec", type=float, default=1.0)
    return parser.parse_args()

def transform_coord_vec(coords, theta, center):
    coords_rel = coords - center
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, s], [-s, c]])
    return np.dot(coords_rel, rot_mat.T)

def compute_extra_features_vec(rel_xy, rot_v_agent, rot_a_agent, rot_v_target, args):
    # 💡 여기서 rel_xy는 Neighbor - Ego의 실시간 거리입니다.
    dx = rel_xy[:, 0]
    dy = rel_xy[:, 1]
    
    # 상대 속도 (Neighbor_v - Ego_v)
    dvx = rot_v_agent[:, 0] - rot_v_target[:, 0]
    dvy = rot_v_agent[:, 1] - rot_v_target[:, 1]
    
    # Neighbor의 가속도 (절대 가속도 유지)
    ax = rot_a_agent[:, 0]
    ay = rot_a_agent[:, 1]

    # lc_state (Neighbor가 Ego를 기준으로 어떻게 움직이는지)
    lc_state = np.zeros_like(dy)
    mask_l = dy < -1.0 # 내 왼쪽 차선에 있을 때
    lc_state[mask_l & (dvy > args.vy_eps)] = -1.0
    lc_state[mask_l & (dvy < -args.vy_eps)] = -3.0
    lc_state[mask_l & (np.abs(dvy) <= args.vy_eps)] = -2.0

    mask_r = dy > 1.0 # 내 오른쪽 차선에 있을 때
    lc_state[mask_r & (dvy < -args.vy_eps)] = 1.0
    lc_state[mask_r & (dvy > args.vy_eps)] = 3.0
    lc_state[mask_r & (np.abs(dvy) <= args.vy_eps)] = 2.0

    # dx_time (충돌 시간/접근 시간)
    denom = dvx.copy()
    denom[dvx >= 0] += args.eps_gate
    denom[dvx < 0] -= args.eps_gate
    dx_time = dx / denom

    # gate (근접성 마스크)
    gate = np.zeros_like(dx_time)
    gate[(-args.t_back < dx_time) & (dx_time < args.t_front)] = 1.0

    # 💡 9개 채널 구성 [dx, dy, dvx, dvy, ax, ay, lc, time, gate]
    return np.stack([dx, dy, dvx, dvy, ax, ay, lc_state, dx_time, gate], axis=-1)

# ==============================================================================
# 2. Main Processing Logic (Pandas 병목 제거)
# ==============================================================================
def process_recording(rec_id: str, raw_dir: Path, temp_dir: Path, args):
    tracks_file = raw_dir / f"{rec_id}_tracks.csv"
    meta_file = raw_dir / f"{rec_id}_tracksMeta.csv"
    rec_meta_file = raw_dir / f"{rec_id}_recordingMeta.csv"
    
    if not (tracks_file.exists() and meta_file.exists() and rec_meta_file.exists()):
        return rec_id, {}, {}

    # 1. 데이터 로드 및 전처리
    df = pd.read_csv(tracks_file)
    tmeta = pd.read_csv(meta_file)
    rmeta = pd.read_csv(rec_meta_file)
    
    raw_fps = rmeta.loc[0, "frameRate"]
    ds_stride = int(round(raw_fps / TARGET_FPS))
    
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id", how="left")
    
    # 차량 중심점 설정
    df["x"] = df["x"] + df["width"] / 2.0
    df["y"] = df["y"] + df["height"] / 2.0
    
    # Upper 방향 차량 좌표 Flip (Driving Direction 1인 경우)
    upper_mask = df["drivingDirection"] == 1
    if upper_mask.any():
        up_m = [float(x) for x in str(rmeta.loc[0, "upperLaneMarkings"]).split(";") if x]
        lo_m = [float(x) for x in str(rmeta.loc[0, "lowerLaneMarkings"]).split(";") if x]
        if up_m and lo_m:
            C_y = up_m[-1] + lo_m[0]
            x_max = df["x"].max()
            df.loc[upper_mask, "x"] = x_max - df.loc[upper_mask, "x"]
            df.loc[upper_mask, "y"] = C_y - df.loc[upper_mask, "y"]
            df.loc[upper_mask, "xVelocity"] *= -1
            df.loc[upper_mask, "yVelocity"] *= -1
            df.loc[upper_mask, "xAcceleration"] *= -1
            df.loc[upper_mask, "yAcceleration"] *= -1
            rmeta.at[0, "upperLaneMarkings"] = ";".join(map(str, [C_y - y for y in up_m][::-1]))
    
    # 다운샘플링 및 정렬
    df = df[(df["frame"] % ds_stride) == 0].sort_values(["id", "frame"]).reset_index(drop=True)

    agents_data = {}
    for vid, group in df.groupby("id"):
        agents_data[vid] = {
            "frames": group["frame"].to_numpy(),
            "data": group[["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].to_numpy()
        }
    
    lifespans = {vid: (info["frames"][0], info["frames"][-1]) for vid, info in agents_data.items()}

    # 차선(Lane) 정보 생성 (Map 데이터용)
    lanes_y = []
    for col in ["upperLaneMarkings", "lowerLaneMarkings"]:
        if col in rmeta.columns and pd.notna(rmeta.loc[0, col]):
            lanes_y.extend([float(x) for x in str(rmeta.loc[0, col]).split(";") if x])
    lanes_y = sorted(list(set(lanes_y)))

    lane_segments = []
    lane_id2idx = {}
    for idx, ly in enumerate(lanes_y):
        pts_x = np.linspace(-1000, 1000, LANE_PTS)
        pts_y = np.full_like(pts_x, ly)
        lane_segments.append(np.stack([pts_x, pts_y, np.zeros(LANE_PTS), np.zeros(LANE_PTS), np.zeros(LANE_PTS)], axis=-1))
        lane_id2idx[str(idx)] = idx
        
    city_name = f"HIGHD_{rec_id}"
    map_dict = {city_name: np.array(lane_segments)}
    global_lane_id2idx = {city_name: lane_id2idx}

    out_name, out_city, out_hist, out_fut, out_lane_id = [], [], [], [], []
    out_norm, out_theta, out_pos, out_valid = [], [], [], []

    # 💡 실험 설정 로드
    extra_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    num_extra = len(extra_indices)
    timestamps = np.arange(-T_H + 1, 1, 1, dtype=np.float32) * (1.0 / TARGET_FPS)
    slide_step = int(round(args.slide_window_sec * TARGET_FPS))

    # 2. Sliding Window 루프 (Ego-Centric 데이터 생성)
    for vid, ego_info in agents_data.items():
        ego_frames = ego_info["frames"]
        ego_data = ego_info["data"]
        if len(ego_frames) < SEQ_LEN: continue
            
        for start_idx in range(0, len(ego_frames) - SEQ_LEN + 1, slide_step):
            obs_idx = start_idx + T_H - 1 
            start_frame = ego_frames[start_idx]
            obs_frame = ego_frames[obs_idx]
            end_frame = ego_frames[start_idx + SEQ_LEN - 1]
            
            if (end_frame - start_frame) != (SEQ_LEN - 1) * ds_stride: continue
            
            # 💡 [기준점 설정] norm_center: Ego의 현재 시점(t=0) 위치
            norm_center = ego_data[obs_idx, :2] 
            vx, vy = ego_data[obs_idx, 2:4]
            theta = np.arctan2(vy, vx)
            
            # 💡 [Ego 데이터] 자기 자신의 과거/미래 궤적
            ego_hist_rel = transform_coord_vec(ego_data[start_idx : obs_idx + 1, :2], theta, norm_center)
            ego_fut_rel = transform_coord_vec(ego_data[obs_idx + 1 : start_idx + SEQ_LEN, :2], theta, norm_center)
            rot_target_vels = transform_coord_vec(ego_data[start_idx : obs_idx + 1, 2:4], theta, np.array([0, 0]))
            rot_target_accs = transform_coord_vec(ego_data[start_idx : obs_idx + 1, 4:6], theta, np.array([0, 0]))
            
            # 텐서 초기화: [0:pos_x, 1:pos_y, 2:time, 3:mask, 4...:extra]
            hist_tensor = np.zeros((MAX_AGENTS, T_H, 4 + num_extra), dtype=np.float32) 
            fut_tensor = np.zeros((MAX_AGENTS, T_F, 3), dtype=np.float32)  
            pos_tensor = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
            
            # 💡 [Index 0] Ego 정보 채우기
            hist_tensor[0, :, 0:2] = ego_hist_rel # 자기 궤적 (x, y)
            hist_tensor[0, :, 2]   = timestamps   # 동기화된 시간 t
            hist_tensor[0, :, 3]   = 1.0          # 마스크 m
            if num_extra > 0:
                # Ego는 자신과의 거리가 항상 0이므로 0행렬 전달
                full_9_ego = compute_extra_features_vec(np.zeros_like(ego_hist_rel), rot_target_vels, rot_target_accs, rot_target_vels, args)
                hist_tensor[0, :, 4:] = full_9_ego[:, extra_indices]
                
            fut_tensor[0, :, :2] = ego_fut_rel
            fut_tensor[0, :, 2] = 1.0 
            pos_tensor[0] = ego_hist_rel[-1]
            
            # 💡 [Index 1~8] Neighbors 정보 채우기
            agent_count = 1
            for nbr_id, nbr_info in agents_data.items():
                if nbr_id == vid or agent_count >= MAX_AGENTS: continue
                if lifespans[nbr_id][1] < start_frame or lifespans[nbr_id][0] > end_frame: continue
                
                nbr_frames = nbr_info["frames"]
                valid_mask = (nbr_frames >= start_frame) & (nbr_frames <= end_frame)
                if not valid_mask.any(): continue
                
                match_frames = nbr_frames[valid_mask]
                match_data = nbr_info["data"][valid_mask]
                t_indices = ((match_frames - start_frame) // ds_stride).astype(int)
                
                # History (실시간 상대 거리 dx, dy 계산)
                h_mask = t_indices < T_H
                if h_mask.any():
                    idx_h = t_indices[h_mask]
                    data_h = match_data[h_mask]
                    
                    # 1) Neighbor의 지도 원점 기준 위치
                    nbr_rel_to_origin = transform_coord_vec(data_h[:, :2], theta, norm_center)
                    
                    # 2) 💡 [실시간 상대 거리] Neighbor(t) - Ego(t)
                    instant_dx_dy = nbr_rel_to_origin - ego_hist_rel[idx_h]
                    
                    hist_tensor[agent_count, idx_h, 0:2] = instant_dx_dy # 상대 거리 (dx, dy)
                    hist_tensor[agent_count, idx_h, 2]   = timestamps[idx_h]
                    hist_tensor[agent_count, idx_h, 3]   = 1.0 
                    
                    if num_extra > 0:
                        rot_v_agent = transform_coord_vec(data_h[:, 2:4], theta, np.array([0, 0]))
                        rot_a_agent = transform_coord_vec(data_h[:, 4:6], theta, np.array([0, 0]))
                        # 실시간 거리를 기반으로 extra feature(lc_state, dx_time 등) 계산
                        full_9_nbr = compute_extra_features_vec(instant_dx_dy, rot_v_agent, rot_a_agent, rot_target_vels[idx_h], args)
                        hist_tensor[agent_count, idx_h, 4:] = full_9_nbr[:, extra_indices]
                        
                    if T_H - 1 in idx_h:
                        pos_tensor[agent_count] = instant_dx_dy[np.where(idx_h == T_H - 1)[0][0]]
                
                # Future (단순 위치 저장)
                f_mask = t_indices >= T_H
                if f_mask.any():
                    idx_f = t_indices[f_mask] - T_H
                    data_f = match_data[f_mask]
                    nbr_fut_rel = transform_coord_vec(data_f[:, :2], theta, norm_center)
                    # Future는 모델이 정답으로 쓰는 것이므로 Ego와의 상대 거리보다는 지도상의 좌표를 유지합니다.
                    fut_tensor[agent_count, idx_f, :2] = nbr_fut_rel
                    fut_tensor[agent_count, idx_f, 2] = 1.0 
                
                if h_mask.any() or f_mask.any(): agent_count += 1
            
            # 최종 리스트 저장
            lane_ids = list(lane_id2idx.values())[:MAX_LANES]
            valid_lane_num = len(lane_ids)
            padded_lane_ids = lane_ids + [-1] * (MAX_LANES - valid_lane_num)
            
            out_name.append(f"{rec_id}_{vid}_{obs_frame}")
            out_city.append(city_name)
            out_hist.append(hist_tensor)
            out_fut.append(fut_tensor)
            out_lane_id.append(padded_lane_ids)
            out_norm.append(norm_center)
            out_theta.append(theta)
            out_pos.append(pos_tensor)
            out_valid.append([agent_count, valid_lane_num])

    # 3. 임시 파일 저장 (병렬 처리 효율성)
    if out_hist:
        temp_file = temp_dir / f"{rec_id}.h5"
        dt_str = h5py.string_dtype(encoding='utf-8')
        with h5py.File(temp_file, 'w') as f:
            f.create_dataset('NAME', data=np.array(out_name, dtype=object), dtype=dt_str)
            f.create_dataset('CITY_NAME', data=np.array(out_city, dtype=object), dtype=dt_str)
            f.create_dataset('HISTORY', data=np.array(out_hist, dtype=np.float32), compression="gzip")
            f.create_dataset('FUTURE', data=np.array(out_fut, dtype=np.float32), compression="gzip")
            f.create_dataset('LANE_ID', data=np.array(out_lane_id, dtype=np.int32), compression="gzip")
            f.create_dataset('NORM_CENTER', data=np.array(out_norm, dtype=np.float32), compression="gzip")
            f.create_dataset('THETA', data=np.array(out_theta, dtype=np.float32), compression="gzip")
            f.create_dataset('POS', data=np.array(out_pos, dtype=np.float32), compression="gzip")
            f.create_dataset('VALID_LEN', data=np.array(out_valid, dtype=np.int32), compression="gzip")

    return rec_id, map_dict, global_lane_id2idx

def process_recording_wrapper(args_tuple): 
    try:
        return process_recording(*args_tuple)
    except Exception as e:
        print(f"\n[Error in Rec {args_tuple[0]}]: {e}")
        traceback.print_exc()
        return None, {}, {}

def balanced_recording_split(ds_counts: dict, ratios=(0.7, 0.1, 0.2), seed=42):
    rng = np.random.default_rng(seed)
    total_samples = sum(ds_counts.values())

    targets = [total_samples * r for r in ratios]
    items = list(ds_counts.items()) 
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True) 

    splits = {"train": [], "val": [], "test": []}
    sums = {"train": 0, "val": 0, "test": 0}
    keys = ["train", "val", "test"]
   
    for rec_id, cnt in items:
        deficits = {k: (targets[j] - sums[k]) for j, k in enumerate(keys)}
        best = max(deficits.items(), key=lambda kv: kv[1])[0]
        splits[best].append(rec_id)
        sums[best] += cnt

    return splits, sums

def merge_h5_files(file_list, out_file):
    if not file_list: return

    total_rows = 0
    shapes, dtypes = {}, {}

    with h5py.File(file_list[0], 'r') as f_sample:
        for k in f_sample.keys():
            shapes[k] = f_sample[k].shape[1:]
            dtypes[k] = f_sample[k].dtype

    for f_path in file_list:
        with h5py.File(f_path, 'r') as f:
            total_rows += f['HISTORY'].shape[0]           

    if total_rows == 0: return

    with h5py.File(out_file, 'w') as out_f:
        dsets = {}
        for k in shapes.keys():
            if dtypes[k].kind == 'O': 
                dsets[k] = out_f.create_dataset(k, shape=(total_rows,) + shapes[k], dtype=dtypes[k])
            else:
                dsets[k] = out_f.create_dataset(k, shape=(total_rows,) + shapes[k], dtype=dtypes[k], compression="gzip")
        
        current_idx = 0
        for f_path in file_list:
            with h5py.File(f_path, 'r') as in_f:
                n = in_f['HISTORY'].shape[0]
                if n == 0: continue
                for k in shapes.keys():
                    dsets[k][current_idx : current_idx + n] = in_f[k][:]
                current_idx += n

def main():
    args = parse_args()
    
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir) / args.feature_mode 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rec_ids = sorted(set([re.match(r"(\d+)_tracks\.csv$", p.name).group(1) for p in raw_dir.glob("*_tracks.csv") if re.match(r"(\d+)_tracks\.csv$", p.name)]))
    
    temp_dir = out_dir / "temp_records"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting multiprocessing with {os.cpu_count()} cores for feature_mode: {args.feature_mode}")
    print(f"Data will be saved to: {out_dir}") # 확인용 출력 추가
    
    print(f"Starting multiprocessing with {os.cpu_count()} cores for feature_mode: {args.feature_mode}")
    final_map_dict = {}
    final_lane_id2idx = {}
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_recording_wrapper, [(rec_id, raw_dir, temp_dir, args) for rec_id in rec_ids], chunksize=1), total=len(rec_ids), desc="Processing HighD"))
        
        for res in results:
            if res[0] is not None:
                final_map_dict.update(res[1])
                final_lane_id2idx.update(res[2])

    ds_counts = {}
    for h5_file in temp_dir.glob("*.h5"):
        rec_id = h5_file.stem
        with h5py.File(h5_file, 'r') as f:
            ds_counts[rec_id] = len(f['HISTORY'])
            
    if not ds_counts:
        print("Error: No data processed.")
        return

    splits, sums = balanced_recording_split(ds_counts, ratios=(0.7, 0.1, 0.2), seed=42)
    print(f"Split results (Sample counts): Train: {sums['train']}, Val: {sums['val']}, Test: {sums['test']}")
    
    for split_name, split_rec_ids in splits.items():
        if not split_rec_ids: continue
        print(f"Merging {split_name} set into {out_dir} ...")
        file_list = [temp_dir / f"{rec_id}.h5" for rec_id in split_rec_ids]
        out_file = out_dir / f"{split_name}.h5"
        merge_h5_files(file_list, out_file)
        print(f"-> Saved {out_file}")

    with open(out_dir / "map.pkl", "wb") as f:
        pickle.dump({"Map": final_map_dict, "Lane_id2idx": final_lane_id2idx}, f)
        print(f"-> Saved Map Data to {out_dir / 'map.pkl'}")

    print("All processing and splitting done!")

if __name__ == "__main__":
    main()