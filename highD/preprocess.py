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

EXTRA_FEATURE_MAP = {
    'baseline': [],
    'exp1': [4, 5],          # lc_state, dx_time
    'exp2': [2, 3, 4, 5, 6], # ax, ay, lc_state, dx_time, gate
    'exp3': [0, 1, 2, 3, 6], # dvx, dvy, ax, ay, gate
    'exp4': [0, 1, 2, 3, 4, 5, 6] # All
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="highD/raw")
    parser.add_argument("--out_dir", type=str, default="highD") 
    parser.add_argument("--feature_mode", type=str, choices=['baseline', 'exp1', 'exp2', 'exp3', 'exp4'], default='baseline')
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
    dx = rel_xy[:, 0]
    dy = rel_xy[:, 1]
    dvx = rot_v_agent[:, 0] - rot_v_target[:, 0]
    dvy = rot_v_agent[:, 1] - rot_v_target[:, 1]
    ax = rot_a_agent[:, 0]
    ay = rot_a_agent[:, 1]

    lc_state = np.zeros_like(dy)
    
    # Left Lane Change (-1, -2, -3)
    mask_l = dy < -1.0
    lc_state[mask_l & (dvy > args.vy_eps)] = -1.0
    lc_state[mask_l & (dvy < -args.vy_eps)] = -3.0
    lc_state[mask_l & (np.abs(dvy) <= args.vy_eps)] = -2.0

    # Right Lane Change (1, 2, 3)
    mask_r = dy > 1.0
    lc_state[mask_r & (dvy < -args.vy_eps)] = 1.0
    lc_state[mask_r & (dvy > args.vy_eps)] = 3.0
    lc_state[mask_r & (np.abs(dvy) <= args.vy_eps)] = 2.0

    denom = dvx.copy()
    denom[dvx >= 0] += args.eps_gate
    denom[dvx < 0] -= args.eps_gate
    dx_time = dx / denom

    gate = np.zeros_like(dx_time)
    gate[(-args.t_back < dx_time) & (dx_time < args.t_front)] = 1.0

    return np.stack([dvx, dvy, ax, ay, lc_state, dx_time, gate], axis=-1)

# ==============================================================================
# 2. Main Processing Logic (Pandas 병목 제거)
# ==============================================================================
def process_recording(rec_id: str, raw_dir: Path, temp_dir: Path, args):
    tracks_file = raw_dir / f"{rec_id}_tracks.csv"
    meta_file = raw_dir / f"{rec_id}_tracksMeta.csv"
    rec_meta_file = raw_dir / f"{rec_id}_recordingMeta.csv"
    
    if not (tracks_file.exists() and meta_file.exists() and rec_meta_file.exists()):
        return rec_id, {}, {}

    df = pd.read_csv(tracks_file)
    tmeta = pd.read_csv(meta_file)
    rmeta = pd.read_csv(rec_meta_file)
    
    raw_fps = rmeta.loc[0, "frameRate"]
    ds_stride = int(round(raw_fps / TARGET_FPS))
    
    df = df.merge(tmeta[["id", "drivingDirection"]], on="id", how="left")
    
    df["x"] = df["x"] + df["width"] / 2.0
    df["y"] = df["y"] + df["height"] / 2.0
    
    # Upper Flip
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
    
    # Downsampling
    df = df[(df["frame"] % ds_stride) == 0].sort_values(["id", "frame"]).reset_index(drop=True)

    agents_data = {}
    for vid, group in df.groupby("id"):
        agents_data[vid] = {
            "frames": group["frame"].to_numpy(),
            "data": group[["x", "y", "xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]].to_numpy()
        }
    
    # 각 에이전트의 생존 프레임 캐싱 (빠른 주변 차량 필터링용)
    lifespans = {vid: (info["frames"][0], info["frames"][-1]) for vid, info in agents_data.items()}

    # 차선 정보 추출
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

    extra_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    num_extra = len(extra_indices)
    timestamps = np.arange(-T_H + 1, 1, 1, dtype=np.float32) * (1.0 / TARGET_FPS)

    slide_step = int(round(args.slide_window_sec * TARGET_FPS))

    # --------------------------------------------------------------------------
    # Sliding Window 
    # --------------------------------------------------------------------------
    for vid, ego_info in agents_data.items():
        ego_frames = ego_info["frames"]
        ego_data = ego_info["data"]
        
        if len(ego_frames) < SEQ_LEN: continue
            
        for start_idx in range(0, len(ego_frames) - SEQ_LEN + 1, slide_step):
            obs_idx = start_idx + T_H - 1 
            start_frame = ego_frames[start_idx]
            obs_frame = ego_frames[obs_idx]
            end_frame = ego_frames[start_idx + SEQ_LEN - 1]
            
            # 연속된 프레임인지 확인 (결측치 방지)
            if (end_frame - start_frame) != (SEQ_LEN - 1) * ds_stride:
                continue
            
            norm_center = ego_data[obs_idx, :2] 
            vx, vy = ego_data[obs_idx, 2:4]
            theta = np.arctan2(vy, vx)
            
            # Target Agent 데이터 변환 (Vectorized)
            ego_hist_rel = transform_coord_vec(ego_data[start_idx : obs_idx + 1, :2], theta, norm_center)
            ego_fut_rel = transform_coord_vec(ego_data[obs_idx + 1 : start_idx + SEQ_LEN, :2], theta, norm_center)
            rot_target_vels = transform_coord_vec(ego_data[start_idx : obs_idx + 1, 2:4], theta, np.array([0, 0]))
            rot_target_accs = transform_coord_vec(ego_data[start_idx : obs_idx + 1, 4:6], theta, np.array([0, 0]))
            
            hist_tensor = np.zeros((MAX_AGENTS, T_H, 4 + num_extra), dtype=np.float32) 
            fut_tensor = np.zeros((MAX_AGENTS, T_F, 3), dtype=np.float32)  
            pos_tensor = np.zeros((MAX_AGENTS, 2), dtype=np.float32)
            
            # Index 0 (Target Agent) 채우기
            hist_tensor[0, :, :2] = ego_hist_rel
            hist_tensor[0, :, 2] = timestamps
            hist_tensor[0, :, 3] = 1.0 
            if num_extra > 0:
                full_extra = compute_extra_features_vec(ego_hist_rel, rot_target_vels, rot_target_accs, rot_target_vels, args)
                hist_tensor[0, :, 4:] = full_extra[:, extra_indices]
                
            fut_tensor[0, :, :2] = ego_fut_rel
            fut_tensor[0, :, 2] = 1.0 
            pos_tensor[0] = ego_hist_rel[-1]
            
            agent_count = 1
            # O(1) 수준으로 시간대가 겹치는 주변 차량만 솎아내기
            for nbr_id, nbr_info in agents_data.items():
                if nbr_id == vid or agent_count >= MAX_AGENTS: continue
                if lifespans[nbr_id][1] < start_frame or lifespans[nbr_id][0] > end_frame: continue
                
                nbr_frames = nbr_info["frames"]
                valid_mask = (nbr_frames >= start_frame) & (nbr_frames <= end_frame)
                if not valid_mask.any(): continue
                
                match_frames = nbr_frames[valid_mask]
                match_data = nbr_info["data"][valid_mask]
                
                # 프레임을 0~39 사이의 인덱스로 매핑
                t_indices = ((match_frames - start_frame) // ds_stride).astype(int)
                
                # History 파트
                h_mask = t_indices < T_H
                if h_mask.any():
                    idx_h = t_indices[h_mask]
                    data_h = match_data[h_mask]
                    rel_xy = transform_coord_vec(data_h[:, :2], theta, norm_center)
                    hist_tensor[agent_count, idx_h, :2] = rel_xy
                    hist_tensor[agent_count, idx_h, 2] = timestamps[idx_h]
                    hist_tensor[agent_count, idx_h, 3] = 1.0 
                    
                    if num_extra > 0:
                        rot_v_agent = transform_coord_vec(data_h[:, 2:4], theta, np.array([0, 0]))
                        rot_a_agent = transform_coord_vec(data_h[:, 4:6], theta, np.array([0, 0]))
                        full_extra = compute_extra_features_vec(rel_xy, rot_v_agent, rot_a_agent, rot_target_vels[idx_h], args)
                        hist_tensor[agent_count, idx_h, 4:] = full_extra[:, extra_indices]
                        
                    if T_H - 1 in idx_h:
                        pos_tensor[agent_count] = rel_xy[np.where(idx_h == T_H - 1)[0][0]]
                
                # Future 파트
                f_mask = t_indices >= T_H
                if f_mask.any():
                    idx_f = t_indices[f_mask] - T_H
                    data_f = match_data[f_mask]
                    rel_xy = transform_coord_vec(data_f[:, :2], theta, norm_center)
                    fut_tensor[agent_count, idx_f, :2] = rel_xy
                    fut_tensor[agent_count, idx_f, 2] = 1.0 
                
                if h_mask.any() or f_mask.any():
                    agent_count += 1
            
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