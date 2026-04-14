import os
import re
import bisect
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import h5py
from concurrent.futures import ProcessPoolExecutor
import traceback
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# 1. Configuration & Constants
# ==============================================================================
TARGET_FPS = 3.0
T_H = 6
T_F = 15
SEQ_LEN = T_H + T_F
MAX_AGENTS = 9
MAX_LANES  = 6
LANE_PTS   = 10

# 8 neighbor slots (same column order as highD tracks CSV)
NEIGHBOR_COLS_8 = [
    "precedingId",
    "followingId",
    "leftPrecedingId",
    "leftAlongsideId",
    "leftFollowingId",
    "rightPrecedingId",
    "rightAlongsideId",
    "rightFollowingId",
]
K = 8   # number of neighbor slots

# Slot priority for top-N gate tie-breaking: 0 > 2 > 5 > 1 > 4 > 7 > 3 > 6
_TOPN_SLOT_PRIORITY = {s: r for r, s in enumerate([0, 2, 5, 1, 4, 7, 3, 6])}

# Empirical slot weights (mean I per slot)
SLOT_WEIGHTS = [0.4944, 0.0411, 0.0935, 0.0074, 0.0002, 0.5559, 0.0000, 0.1179]

# Conditional slot weights derived from SlotWeightProbe models (mean softmax per slot).
# Used when --slot_importance_conditional is set.

# No-LC case: weights by ego lane level  (0=leftmost/fast, 1=middle, 2=rightmost/slow)
SLOT_WEIGHTS_BY_LANE_LEVEL = [
    [0.4657, 0.0163, 0.0000, 0.0000, 0.0000, 0.4357, 0.0035, 0.0788],  # ll0 leftmost
    [0.4240, 0.0346, 0.3347, 0.0197, 0.1859, 0.0007, 0.0002, 0.0001],  # ll1 middle
    [0.3846, 0.0141, 0.3593, 0.0345, 0.2070, 0.0000, 0.0000, 0.0000],  # ll2 rightmost
]

# LC-in-history case: pre-LC weights per lc_type (0-5)
SLOT_WEIGHTS_PRE_LC = [
    [0.0000, 0.0037, 0.0000, 0.0000, 0.0000, 0.2718, 0.1157, 0.6089],  # lct0 leftmost→middle
    [0.7023, 0.1658, 0.0000, 0.0000, 0.0000, 0.1251, 0.0049, 0.0019],  # lct1 leftmost→rightmost
    [0.3170, 0.0117, 0.0033, 0.0003, 0.0005, 0.5215, 0.0168, 0.1289],  # lct2 middle→leftmost
    [0.0367, 0.0057, 0.4062, 0.1076, 0.4435, 0.0000, 0.0000, 0.0001],  # lct3 middle→rightmost
    [0.9996, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # lct4 rightmost→leftmost
    [0.0048, 0.0000, 0.5762, 0.1229, 0.2962, 0.0000, 0.0000, 0.0000],  # lct5 rightmost→middle
]

# LC-in-history case: post-LC weights per lc_type (0-5)
SLOT_WEIGHTS_POST_LC = [
    [0.0017, 0.0074, 0.0026, 0.0011, 0.0109, 0.4849, 0.0611, 0.4303],  # lct0 leftmost→middle
    [0.0478, 0.0078, 0.7227, 0.0393, 0.1825, 0.0000, 0.0000, 0.0000],  # lct1 leftmost→rightmost
    [0.8647, 0.0680, 0.0000, 0.0000, 0.0000, 0.0527, 0.0042, 0.0103],  # lct2 middle→leftmost
    [0.0557, 0.9204, 0.0001, 0.0001, 0.0237, 0.0000, 0.0000, 0.0000],  # lct3 middle→rightmost
    [0.0002, 0.0001, 0.0000, 0.0000, 0.0000, 0.9557, 0.0427, 0.0013],  # lct4 rightmost→leftmost
    [0.0125, 0.0334, 0.0001, 0.0016, 0.0006, 0.2424, 0.0296, 0.6799],  # lct5 rightmost→middle
]

# (from_level, to_level) → lc_type
_LC_TYPE_MAP_LEVEL: Dict[Tuple[int, int], int] = {
    (0, 1): 0, (0, 2): 1,
    (1, 0): 2, (1, 2): 3,
    (2, 0): 4, (2, 1): 5,
}

# ------------------------------------------------------------------
# Full 13-channel neighbor feature indices
# [0:dx, 1:dy, 2:dvx, 3:dvy, 4:ax, 5:ay, 6:lc_state, 7:volume, 8:size_bin,
#  9:gate, 10:I_x, 11:I_y, 12:I]
# Total full channels = 13
# ------------------------------------------------------------------
# fmt: off
EXTRA_FEATURE_MAP = {
    'baseline': [0, 1, 2, 3, 4, 5],
    'importance': [0, 1, 2, 3, 4, 5, 12],
    'Iy': [0, 1, 2, 3, 4, 5, 11],
    'dimI': [0, 1, 2, 3, 4, 5, 8, 11],
}
# fmt: on

# ==============================================================================
# 2. LIS Binning  (ported from preprocess_neighformer.py)
# ==============================================================================
LIS_BINS: Dict[str, Dict] = {
    '3': {'cuts': [-5.8639,  4.9525],
          'vals': [-1.0, 0.0, 1.0],
          'L': 1.0},
    '5': {'cuts': [-13.7033, -3.0238, 2.2735, 13.0957],
          'vals': [-2.0, -1.0, 0.0, 1.0, 2.0],
          'L': 2.0},
    '7': {'cuts': [-18.7902, -8.2922, -1.9963, 1.3381, 7.3744, 18.5267],
          'vals': [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
          'L': 3.0},
    '9': {'cuts': [-22.7661, -12.1209, -5.8639, -1.4829, 0.9127, 4.9525, 11.4115, 22.7702],
          'vals': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
          'L': 4.0},
}

def _lit_to_lis(lit: float, lis_mode: str) -> float:
    cfg = LIS_BINS[lis_mode]
    return cfg['vals'][bisect.bisect_right(cfg['cuts'], lit)]

# ==============================================================================
# 3. Importance Parameters & Computation  (ported from preprocess_neighformer.py)
# ==============================================================================
IMPORTANCE_PARAMS_LIS: Dict[str, float] = {
    'sx': 1.0, 'ax': 0.15, 'bx': 0.2,
    'sy': 2.0, 'ay': 0.1,  'by': 0.1, 'py': 1.5,
}

IMPORTANCE_PARAMS_LIT: Dict[str, float] = {
    'sx': 15.0, 'ax': 0.2, 'bx': 0.25,
    'sy':  2.0, 'ay': 0.01, 'by': 0.1,
}

def compute_importance_lis(
    lis: float, delta_lane: float, lc_state: float
) -> Tuple[float, float, float]:
    """
    I_x = exp(-(lis^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lis|^py) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)
    Params: sx=1.0, ax=0.15, bx=0.2, sy=2.0, ay=0.1, by=0.1, py=1.5
    """
    p  = IMPORTANCE_PARAMS_LIS
    ix = float(np.exp(-(lis ** 2) / (2.0 * p["sx"] ** 2))
               * np.exp(-p["ax"] * lc_state)
               * np.exp(-p["bx"] * delta_lane))
    iy = float(np.exp(-(lc_state ** 2) / (2.0 * p["sy"] ** 2))
               * np.exp(-p["ay"] * (abs(lis) ** p["py"]))
               * np.exp(-p["by"] * delta_lane))
    i_total = float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))
    return ix, iy, i_total


def compute_importance_lit(
    lit: float, delta_lane: float, lc_state: float
) -> Tuple[float, float, float]:
    """
    I_x = exp(-(lit^2 / (2*sx^2))) * exp(-ax * lc_state) * exp(-bx * delta_lane)
    I_y = exp(-(lc_state^2 / (2*sy^2))) * exp(-ay * |lit|^1.5) * exp(-by * delta_lane)
    I   = sqrt((I_x^2 + I_y^2) / 2)
    Params: sx=15.0, ax=0.2, bx=0.25, sy=2.0, ay=0.01, by=0.1
    """
    p  = IMPORTANCE_PARAMS_LIT
    ix = float(np.exp(-(lit ** 2) / (2.0 * p["sx"] ** 2))
               * np.exp(-p["ax"] * lc_state)
               * np.exp(-p["bx"] * delta_lane))
    iy = float(np.exp(-(lc_state ** 2) / (2.0 * p["sy"] ** 2))
               * np.exp(-p["ay"] * (abs(lit) ** 1.5))
               * np.exp(-p["by"] * delta_lane))
    i_total = float(np.sqrt((ix ** 2 + iy ** 2) / 2.0))
    return ix, iy, i_total

# ==============================================================================
# 4. Top-N gate helper  (ported from preprocess_neighformer.py)
# ==============================================================================
def _apply_topn_gate(nb_row: np.ndarray, mask_row: np.ndarray, n: int) -> None:
    """Select top-n slots by I (idx 12) and zero-gate the rest (in-place)."""
    K_local = nb_row.shape[0]
    valid = [k for k in range(K_local) if mask_row[k]]
    valid.sort(key=lambda k: (-nb_row[k, 12], _TOPN_SLOT_PRIORITY.get(k, K_local)))
    selected = set(valid[:n])
    for k in valid:
        if k not in selected:
            nb_row[k, 9]  = 0.0
            nb_row[k, 10] = 0.0
            nb_row[k, 11] = 0.0
            nb_row[k, 12] = 0.0


def _lane_id_to_level(lid: int, dd: int, sorted_lids: List[int], post_flip: bool) -> int:
    """lane_id → lane_level (0=leftmost/fast, 1=middle, 2=rightmost/slow)."""
    n = len(sorted_lids)
    if n == 0 or lid not in sorted_lids:
        return -1
    idx = sorted_lids.index(lid)
    if n == 1:
        return 1
    if post_flip or dd == 2:
        if idx == 0:     return 0
        if idx == n - 1: return 2
        return 1
    else:  # dd=1, no flip
        if idx == 0:     return 2
        if idx == n - 1: return 0
        return 1


def _ego_lc_context(
    ego_lane_arr: np.ndarray,
    dd: int,
    lane_ids_per_dd: Dict[int, List[int]],
    post_flip: bool,
) -> Tuple[int, Optional[int], int]:
    """history window 내 ego LC 상태를 판단한다.

    Returns (lane_level, lc_frame_ti, lc_type)
      lane_level  : 0/1/2 (no-LC, ego의 t0 차선), -2 (LC in history), -1 (unknown)
      lc_frame_ti : LC가 처음 일어난 hist frame 인덱스 (None = no LC)
      lc_type     : 0-5  (-1 = no LC or unknown)
    """
    sorted_lids = lane_ids_per_dd.get(dd, [])
    lc_frame_ti: Optional[int] = None
    lc_type = -1
    for ti in range(1, len(ego_lane_arr)):
        if ego_lane_arr[ti] != ego_lane_arr[ti - 1]:
            lc_frame_ti = ti
            from_lvl = _lane_id_to_level(int(ego_lane_arr[ti - 1]), dd, sorted_lids, post_flip)
            to_lvl   = _lane_id_to_level(int(ego_lane_arr[ti]),     dd, sorted_lids, post_flip)
            lc_type  = _LC_TYPE_MAP_LEVEL.get((from_lvl, to_lvl), -1)
            break
    if lc_frame_ti is None:
        lane_level = _lane_id_to_level(int(ego_lane_arr[-1]), dd, sorted_lids, post_flip)
    else:
        lane_level = -2
    return lane_level, lc_frame_ti, lc_type


def _get_slot_weight(
    ki: int,
    ti: int,
    lane_level: int,
    lc_frame_ti: Optional[int],
    lc_type: int,
) -> float:
    """slot ki / timestep ti에 대응하는 조건부 slot weight를 반환."""
    if lc_frame_ti is not None and lc_type >= 0:
        if ti < lc_frame_ti:
            return SLOT_WEIGHTS_PRE_LC[lc_type][ki]
        else:
            return SLOT_WEIGHTS_POST_LC[lc_type][ki]
    elif 0 <= lane_level <= 2:
        return SLOT_WEIGHTS_BY_LANE_LEVEL[lane_level][ki]
    else:
        return SLOT_WEIGHTS[ki]  # fallback


# ==============================================================================
# 5. Argument Parsing
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",      type=str,   default="highD/raw")
    parser.add_argument("--out_dir",      type=str,   default="highD")
    parser.add_argument("--feature_mode", type=str,   choices=EXTRA_FEATURE_MAP.keys(), default='baseline')
    parser.add_argument("--slide_window_sec", type=float, default=1.0)

    # lc / gating (aligned with neighformer defaults)
    parser.add_argument("--t_front",       type=float, default=3.0)
    parser.add_argument("--t_back",        type=float, default=5.0)
    parser.add_argument("--vy_eps",        type=float, default=0.27,
                        help="yV threshold for lc_version=v1")
    parser.add_argument("--eps_gate",      type=float, default=1.0,
                        help="eps for LIT denominator clamp (raised to 1.0 to match neighformer)")
    parser.add_argument("--dvy_eps_cross", type=float, default=0.26,
                        help="lc_state v2: |dvy| threshold for cross-lane slot neighbors")
    parser.add_argument("--dvy_eps_same",  type=float, default=1.03,
                        help="lc_state v2: |dvy| threshold for same-lane slot (0/1) neighbors")
    parser.add_argument("--dy_same",       type=float, default=1.5,
                        help="lc_state v2: |dy| < dy_same means same-lane for slot 0/1")

    # LIS
    parser.add_argument("--lis_mode", default="7", choices=["3", "5", "7", "9"],
                        help="LIS binning mode: 3={-1,0,1} | 5={-2..2} | 7={-3..3} | 9={-4..4}")

    # importance
    parser.add_argument("--importance_mode", default="lis", choices=["lis", "lit"],
                        help="lis=use discrete LIS | lit=use continuous LIT (legacy params)")

    # gate
    parser.add_argument("--gate_theta", type=float, default=0.0,
                        help="I threshold gate: gate=1 if I>=theta. 0.0=all active")
    parser.add_argument("--gate_topn",  type=int,   default=0,
                        help="Top-N gate: keep up to N slots with highest I. 0=disabled")
    parser.add_argument("--gate_mask",  action="store_true", default=False,
                        help="If set, gate=0 neighbors are zeroed in hist_tensor and excluded from valid count")
    parser.add_argument("--slot_importance", type=float, default=0.0, dest="slot_importance_alpha",
                        help="Slot importance boost: I_new = min(I*(1+alpha*w_slot), 1.0). 0.0=disabled")
    parser.add_argument("--slot_importance_conditional", action="store_true", default=False,
                        help="use lane-level/pre-LC/post-LC conditional slot weights")

    # lc_state version
    parser.add_argument("--lc_version", default="v4", choices=["v1", "v2", "v3", "v4"],
                        help="lc_state 계산 방식: "
                             "v1=slot기반 절대yV | v2=dvy기반+slot/dy조합 | "
                             "v3=latV+lco기반 (default) | v4=lco_norm기반")

    return parser.parse_args()

# ==============================================================================
# 6. Coordinate Transform (rotation, used for ego-centric frame)
# ==============================================================================
def transform_coord_vec(coords, theta, center):
    coords_rel = coords - center
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, s], [-s, c]])
    return np.dot(coords_rel, rot_mat.T)

# ==============================================================================
# 7. Vehicle size bin  (ported from neighformer preprocess.py)
# ==============================================================================
# Bin edges for width * length * height_est (m³): [12, 20, 90, 150]
# Bin values: 0 (소형차) ~ 4 (대형 트럭)
_VOLUME_BIN_EDGES = [12.0, 20.0, 90.0, 150.0]  # 4 inner cuts → 5 bins


def _volume_bin(phys_length: float, phys_width: float, vehicle_class: str) -> Tuple[float, float]:
    """Return (size bin index 0~4, raw volume m³) for a vehicle.

    height is estimated from vehicle class and physical length:
      Car:   length < 4.5m → 1.45m,  < 5.0m → 1.70m,  >= 5.0m → 1.90m
      Truck: length < 12.0m → 2.75m, >= 12.0m → 3.75m
    """
    if vehicle_class == "Car":
        if phys_length < 4.5:   height = 1.45
        elif phys_length < 5.0: height = 1.70
        else:                   height = 1.90
    else:
        height = 2.75 if phys_length < 12.0 else 3.75
    volume = phys_width * phys_length * height
    for i, edge in enumerate(_VOLUME_BIN_EDGES):
        if volume < edge:
            return float(i), volume
    return 4.0, volume


# ==============================================================================
# 8. Per-neighbor feature computation
#    Full 13-channel output:
#    [dx, dy, dvx, dvy, ax, ay, lc_state, volume, size_bin, gate, I_x, I_y, I]
#
#    - volume: physical vehicle volume  width * length * height_est  (m³)
#    - size_bin: vehicle size bin (0~4) based on volume
#    - LIT/LIS are computed internally for importance only (not stored)
#    - Importance: I_x, I_y, I computed from (lis|lit), delta_lane, lc_state
#    - gate: based on I threshold OR top-N (post-processed)
#    - lc_state: v1~v4 selectable (v4 default, matching neighformer)
#
#    Called once per (neighbor slot, timestep).
# ==============================================================================
def compute_nb_features_scalar(
    dx: float, dy: float,
    dvx: float, dvy: float,
    ax: float, ay: float,
    nb_yv: float,          # neighbor absolute lateral velocity (for v3/v4)
    nb_lco: float,         # neighbor lateral lane-center offset (for v3/v4)
    nb_lco_norm: float,    # neighbor lco / (lane_width*0.5)   (for v4)
    ki: int,               # slot index 0..7
    len_ego: float,        # ego vehicle length (width in highD = longitudinal)
    len_nb: float,         # neighbor vehicle length
    delta_lane: int,       # |nb_lane_id - ego_lane_id|
    args,
    # per-sample conditional slot weight context (ignored when slot_importance_conditional=False)
    _lc_lane_lv: int = -1,
    _lc_frame_ti: Optional[int] = None,
    _lc_type: int = -1,
    _ti: int = 0,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Returns (lc_state, lit, lis, gate, I_x, I_y, I_total)
    plus the raw (ax, ay) that were passed in.
    """

    # ── lc_state ──────────────────────────────────────────────────────────────
    if args.lc_version == "v1":
        # slot-based absolute yV  (original mmT v1 logic, 7-value range)
        if ki < 2:
            lc_state = 0.0
        elif ki < 5:   # left group
            if   nb_yv >  args.vy_eps:  lc_state = -1.0
            elif nb_yv < -args.vy_eps:  lc_state = -3.0
            else:                       lc_state = -2.0
        else:          # right group
            if   nb_yv < -args.vy_eps:  lc_state =  1.0
            elif nb_yv >  args.vy_eps:  lc_state =  3.0
            else:                       lc_state =  2.0

    elif args.lc_version == "v2":
        # dvy-based + slot/dy combination  {0: closing, 1: stay, 2: moving out}
        abs_dvy = abs(dvy)
        if ki < 2 and abs(dy) < args.dy_same:
            lc_state = 2.0 if abs_dvy > args.dvy_eps_same else 1.0
        elif ki >= 2:
            if abs_dvy > args.dvy_eps_cross:
                lc_state = 0.0 if dy * dvy < 0 else 2.0
            else:
                lc_state = 1.0
        else:
            lc_state = 0.0 if dy * dvy < 0 else 2.0

    elif args.lc_version == "v3":
        # latV + lco based  {0: closing, 1: stay, 2: moving out}
        if ki < 2:   # same-lane (lead / rear)
            if (nb_lco < -1.0 and nb_yv > 0.0) or (nb_lco > 1.0 and nb_yv < 0.0):
                lc_state = 0.0
            elif (nb_lco < -1.0 and nb_yv < 0.0) or (nb_lco > 1.0 and nb_yv > 0.0) \
                    or abs(nb_yv) > 0.029:
                lc_state = 2.0
            else:
                lc_state = 1.0
        elif ki < 5:  # left-lane group (slots 2,3,4)
            if   nb_yv < -0.029: lc_state = 0.0
            elif nb_yv >  0.029: lc_state = 2.0
            else:                lc_state = 1.0
        else:         # right-lane group (slots 5,6,7)
            if   nb_yv < -0.029: lc_state = 2.0
            elif nb_yv >  0.029: lc_state = 0.0
            else:                lc_state = 1.0

    else:  # v4: lco_norm-based boundary detection + slot-direction decision
        if abs(nb_lco_norm) <= 0.5:
            lc_state = 1.0
        elif ki < 2:
            lc_state = 0.0 if nb_lco_norm * nb_yv < 0 else 2.0
        elif ki < 5:
            lc_state = 0.0 if nb_yv < 0 else 2.0
        else:
            lc_state = 0.0 if nb_yv > 0 else 2.0

    # ── LIT (bumper-to-bumper gap, direction-aware)  ────────────────────────
    half_sum = 0.5 * (len_ego + len_nb)
    if dx >= 0:   # nb ahead
        gap        = abs(dx - half_sum)
        denom_base = dvx
    else:         # nb behind
        gap        = abs(-dx - half_sum)
        denom_base = -dvx
    eps = args.eps_gate
    lit = gap / (denom_base + (eps if denom_base >= 0 else -eps))

    # ── LIS ─────────────────────────────────────────────────────────────────
    lis = _lit_to_lis(lit, args.lis_mode)

    # ── Importance ──────────────────────────────────────────────────────────
    if args.importance_mode == "lit":
        ix, iy, i_total = compute_importance_lit(lit, float(delta_lane), lc_state)
    else:
        ix, iy, i_total = compute_importance_lis(lis, float(delta_lane), lc_state)

    # ── Slot importance boost ────────────────────────────────────────────────
    if args.slot_importance_alpha > 0.0:
        if args.slot_importance_conditional:
            w_slot = _get_slot_weight(ki, _ti, _lc_lane_lv, _lc_frame_ti, _lc_type)
        else:
            w_slot = SLOT_WEIGHTS[ki]
        i_total = min(i_total * (1.0 + args.slot_importance_alpha * w_slot), 1.0)

    # ── Gate (threshold; top-N applied later per timestep) ──────────────────
    if args.gate_theta > 0.0:
        gate = 1.0 if i_total >= args.gate_theta else 0.0
    else:
        gate = 1.0

    return lc_state, lit, lis, gate, ix * gate, iy * gate, i_total * gate

# ==============================================================================
# 8. Main Processing Logic
# ==============================================================================
def process_recording(rec_id: str, raw_dir: Path, temp_dir: Path, args):
    tracks_file   = raw_dir / f"{rec_id}_tracks.csv"
    meta_file     = raw_dir / f"{rec_id}_tracksMeta.csv"
    rec_meta_file = raw_dir / f"{rec_id}_recordingMeta.csv"

    if not (tracks_file.exists() and meta_file.exists() and rec_meta_file.exists()):
        return rec_id, {}, {}

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    tracks = pd.read_csv(tracks_file)
    tmeta  = pd.read_csv(meta_file)
    rmeta  = pd.read_csv(rec_meta_file)

    raw_fps   = float(rmeta.loc[0, "frameRate"])
    ds_stride = int(round(raw_fps / TARGET_FPS))

    # Ensure required columns exist
    for c in NEIGHBOR_COLS_8:
        if c not in tracks.columns: tracks[c] = 0
    for c in ["xVelocity", "yVelocity", "xAcceleration", "yAcceleration"]:
        if c not in tracks.columns: tracks[c] = 0.0
    if "laneId" not in tracks.columns: tracks["laneId"] = 0

    # vehicle-level lookups
    vid_to_dd    = dict(zip(tmeta["id"].astype(int), tmeta["drivingDirection"].astype(int)))
    vid_to_w     = dict(zip(tmeta["id"].astype(int), tmeta["width"].astype(float)))   # longitudinal length in highD
    vid_to_h     = dict(zip(tmeta["id"].astype(int), tmeta["height"].astype(float)))  # lateral width
    vid_to_class = dict(zip(tmeta["id"].astype(int), tmeta["class"].astype(str)))     # vehicle class (Car/Truck)

    # ── Raw arrays ───────────────────────────────────────────────────────────
    frame   = tracks["frame"].astype(np.int32).to_numpy()
    vid_arr = tracks["id"].astype(np.int32).to_numpy()
    x       = tracks["x"].astype(np.float32).to_numpy().copy()
    y       = tracks["y"].astype(np.float32).to_numpy().copy()
    w_row   = np.array([vid_to_w.get(int(v), 0.0) for v in vid_arr], np.float32)
    h_row   = np.array([vid_to_h.get(int(v), 0.0) for v in vid_arr], np.float32)
    x      += 0.5 * w_row   # convert bbox corner → center
    y      += 0.5 * h_row
    xv      = tracks["xVelocity"].astype(np.float32).to_numpy()
    yv      = tracks["yVelocity"].astype(np.float32).to_numpy()
    xa      = tracks["xAcceleration"].astype(np.float32).to_numpy()
    ya      = tracks["yAcceleration"].astype(np.float32).to_numpy()
    lane_id = tracks["laneId"].astype(np.int16).to_numpy()
    dd      = np.array([vid_to_dd.get(int(v), 0) for v in vid_arr], np.int8)
    x_max   = float(np.nanmax(x)) if len(x) else 0.0

    # ── Lane markings ────────────────────────────────────────────────────────
    up_m = ([float(p) for p in str(rmeta.loc[0, "upperLaneMarkings"]).split(";") if p]
            if "upperLaneMarkings" in rmeta.columns else [])
    lo_m = ([float(p) for p in str(rmeta.loc[0, "lowerLaneMarkings"]).split(";") if p]
            if "lowerLaneMarkings" in rmeta.columns else [])
    upper_mark = np.array(up_m, np.float32)
    lower_mark = np.array(lo_m, np.float32)
    C_y = float(upper_mark[-1] + lower_mark[0]) if (len(upper_mark) and len(lower_mark)) else 0.0
    _N_upper = len(upper_mark)

    # ── Lateral lane-center offset (pre-flip coordinates) ────────────────────
    # Needed for lc_state v3 / v4
    lat_lane_offset_arr = np.zeros(len(y), np.float32)
    _lid_arr = lane_id.astype(np.int32)

    _mask_lo = (dd == 2)
    _j_lo    = _lid_arr - _N_upper - 2
    _ok_lo   = _mask_lo & (_j_lo >= 0) & (_j_lo < len(lower_mark) - 1)
    lat_lane_offset_arr[_ok_lo] = (
        y[_ok_lo] - 0.5 * (lower_mark[_j_lo[_ok_lo]] + lower_mark[_j_lo[_ok_lo] + 1])
    )

    _mask_up = (dd == 1)
    _j_up    = _lid_arr - 2
    _ok_up   = _mask_up & (_j_up >= 0) & (_j_up < len(upper_mark) - 1)
    lat_lane_offset_arr[_ok_up] = (
        y[_ok_up] - 0.5 * (upper_mark[_j_up[_ok_up]] + upper_mark[_j_up[_ok_up] + 1])
    )
    lat_lane_offset_arr[dd == 1] *= -1.0   # negate to match post-flip sign

    # lane width array (for v4 lco_norm)
    lat_lane_width_arr = np.full(len(y), 3.75, np.float32)
    lat_lane_width_arr[_ok_lo] = np.abs(lower_mark[_j_lo[_ok_lo] + 1] - lower_mark[_j_lo[_ok_lo]])
    lat_lane_width_arr[_ok_up] = np.abs(upper_mark[_j_up[_ok_up] + 1] - upper_mark[_j_up[_ok_up]])

    # ── Upper-direction flip ─────────────────────────────────────────────────
    upper_for_calc = np.sort((C_y - upper_mark).astype(np.float32)) if len(upper_mark) else upper_mark
    # build lane-center / lane-width tables (post-flip, for lane map)
    def _build_lane_tables(markings):
        if markings is None or len(markings) < 2:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        left, right = markings[:-1], markings[1:]
        return ((right + left) * 0.5).astype(np.float32), (right - left).astype(np.float32)

    upper_center, _ = _build_lane_tables(upper_for_calc)
    lower_center, _ = _build_lane_tables(lower_mark)
    upper_mm = (1, int(len(upper_center))) if len(upper_center) else None

    mask_up = (dd == 1)
    if np.any(mask_up):
        x2, y2, xv2, yv2, xa2, ya2, l2 = (a.copy() for a in (x, y, xv, yv, xa, ya, lane_id))
        x2[mask_up]  = x_max - x2[mask_up]
        y2[mask_up]  = C_y   - y2[mask_up]
        xv2[mask_up] = -xv2[mask_up]; yv2[mask_up] = -yv2[mask_up]
        xa2[mask_up] = -xa2[mask_up]; ya2[mask_up] = -ya2[mask_up]
        if upper_mm is not None:
            mn, mx_v = upper_mm
            ok = mask_up & (l2 > 0)
            l2[ok] = (mn + mx_v) - l2[ok]
        x, y, xv, yv, xa, ya, lane_id = x2, y2, xv2, yv2, xa2, ya2, l2

    # ── Global shift (align all coords to x_min, y_min) ─────────────────────
    x_min = float(np.nanmin(x)) if x.size else 0.0
    y_min = float(np.nanmin(y)) if y.size else 0.0
    x = (x - x_min).astype(np.float32)
    y = (y - y_min).astype(np.float32)
    if len(upper_center): upper_center = (upper_center - y_min).astype(np.float32)
    if len(lower_center): lower_center = (lower_center - y_min).astype(np.float32)

    # ── Downsample ───────────────────────────────────────────────────────────
    keep = (frame % ds_stride) == 0
    frame   = frame[keep];   vid_arr = vid_arr[keep]
    x       = x[keep];       y       = y[keep]
    xv      = xv[keep];      yv      = yv[keep]
    xa      = xa[keep];      ya      = ya[keep]
    lane_id = lane_id[keep]
    lat_lane_offset_arr = lat_lane_offset_arr[keep]
    lat_lane_width_arr  = lat_lane_width_arr[keep]

    nb_ids_all = np.stack(
        [tracks[c].astype(np.int32).to_numpy()[keep] for c in NEIGHBOR_COLS_8], axis=1
    )

    # ── Build per-vehicle index structures ───────────────────────────────────
    per_vid_rows: Dict[int, np.ndarray]      = {}
    per_vid_frame_to_row: Dict[int, Dict[int, int]] = {}
    from collections import defaultdict
    _tmp: Dict[int, List[int]] = defaultdict(list)
    for row_i, v in enumerate(vid_arr):
        _tmp[int(v)].append(row_i)
    for v, rows in _tmp.items():
        rows_arr = np.array(rows, np.int32)
        rows_arr = rows_arr[np.argsort(frame[rows_arr])]
        per_vid_rows[v] = rows_arr
        per_vid_frame_to_row[v] = {int(frame[r]): int(r) for r in rows_arr}

    # ── per-dd sorted lane IDs (for conditional slot weights) ────────────────
    lane_ids_per_dd: Dict[int, List[int]] = {}
    if args.slot_importance_conditional:
        for dd_val in [1, 2]:
            lids = sorted(set(int(x) for x in lane_id[dd == dd_val] if x > 0))
            lane_ids_per_dd[dd_val] = lids

    # ── Map data (lane segments) ─────────────────────────────────────────────
    all_lane_y = sorted(set(
        [float(v) for v in upper_center] + [float(v) for v in lower_center]
    ))
    lane_segments = []
    lane_id2idx   = {}
    for idx, ly in enumerate(all_lane_y):
        pts_x = np.linspace(-1000, 1000, LANE_PTS)
        pts_y = np.full_like(pts_x, ly)
        lane_segments.append(np.stack([pts_x, pts_y,
                                       np.zeros(LANE_PTS),
                                       np.zeros(LANE_PTS),
                                       np.zeros(LANE_PTS)], axis=-1))
        lane_id2idx[str(idx)] = idx

    city_name        = f"HIGHD_{rec_id}"
    map_dict         = {city_name: np.array(lane_segments) if lane_segments else np.zeros((0,))}
    global_lane_id2idx = {city_name: lane_id2idx}

    # ── Experiment config ────────────────────────────────────────────────────
    extra_indices = EXTRA_FEATURE_MAP[args.feature_mode]
    num_extra     = len(extra_indices)
    # Full feature vector has 13 channels: [dx,dy,dvx,dvy,ax,ay,lc,lit,lis,gate,Ix,Iy,I]
    FULL_DIM = 13

    timestamps = np.arange(-T_H + 1, 1, 1, dtype=np.float32) * (1.0 / TARGET_FPS)
    slide_step = int(round(args.slide_window_sec * TARGET_FPS))

    out_name, out_city, out_hist, out_fut, out_lane_id = [], [], [], [], []
    out_norm, out_theta, out_pos, out_valid = [], [], [], []
    out_meta_rec, out_meta_track, out_meta_frame = [], [], []

    # ── Sliding window loop ──────────────────────────────────────────────────
    for v, v_rows in per_vid_rows.items():
        frs = frame[v_rows]
        total_frames_needed = SEQ_LEN * ds_stride
        if (int(frs[-1]) - int(frs[0])) < (SEQ_LEN - 1) * ds_stride:
            continue

        fr_set    = set(int(f) for f in frs)
        start_min = int(frs[0]  + (T_H - 1) * ds_stride)
        end_max   = int(frs[-1] - T_F       * ds_stride)
        if start_min > end_max:
            continue

        len_ego = float(vid_to_w.get(v, 0.0))

        obs_frame_val = start_min
        while obs_frame_val <= end_max:
            hist_frames = [obs_frame_val - (T_H - 1 - i) * ds_stride for i in range(T_H)]
            fut_frames  = [obs_frame_val + (i + 1)       * ds_stride for i in range(T_F)]

            if not all(hf in fr_set for hf in hist_frames) or \
               not all(ff in fr_set for ff in fut_frames):
                obs_frame_val += slide_step * ds_stride
                continue

            ego_h_rows = [per_vid_frame_to_row[v][hf] for hf in hist_frames]
            ego_f_rows = [per_vid_frame_to_row[v][ff] for ff in fut_frames]

            # ── Ego-centric normalisation (same as neighformer: last hist frame as origin)
            ref_x = float(x[ego_h_rows[-1]])
            ref_y = float(y[ego_h_rows[-1]])

            ego_xy  = np.stack([x[ego_h_rows] - ref_x, y[ego_h_rows] - ref_y], axis=1).astype(np.float32)
            ego_vxy = np.stack([xv[ego_h_rows], yv[ego_h_rows]], axis=1).astype(np.float32)
            ego_axy = np.stack([xa[ego_h_rows], ya[ego_h_rows]], axis=1).astype(np.float32)

            # rotation from ego heading at obs time (last hist step)
            ego_vx_obs = float(xv[ego_h_rows[-1]])
            ego_vy_obs = float(yv[ego_h_rows[-1]])
            theta = float(np.arctan2(ego_vy_obs, ego_vx_obs))

            # rotate ego history into ego-centric frame
            norm_center  = np.array([ref_x, ref_y], np.float32)
            ego_hist_rel = transform_coord_vec(
                np.stack([x[ego_h_rows], y[ego_h_rows]], axis=1), theta, norm_center
            ).astype(np.float32)
            ego_fut_rel  = transform_coord_vec(
                np.stack([x[ego_f_rows], y[ego_f_rows]], axis=1), theta, norm_center
            ).astype(np.float32)
            rot_ego_vels = transform_coord_vec(
                np.stack([xv[ego_h_rows], yv[ego_h_rows]], axis=1), theta, np.zeros(2)
            ).astype(np.float32)
            rot_ego_accs = transform_coord_vec(
                np.stack([xa[ego_h_rows], ya[ego_h_rows]], axis=1), theta, np.zeros(2)
            ).astype(np.float32)

            ego_lane_arr = lane_id[ego_h_rows].astype(np.int32)   # lane id per hist step

            # ── conditional slot weight context (computed once per sample) ────
            _lc_lane_lv: int            = -1
            _lc_frame_ti: Optional[int] = None
            _lc_type: int               = -1
            if args.slot_importance_conditional and args.slot_importance_alpha > 0.0:
                _ego_dd = vid_to_dd.get(v, 2)
                _lc_lane_lv, _lc_frame_ti, _lc_type = _ego_lc_context(
                    ego_lane_arr, _ego_dd, lane_ids_per_dd, True  # normalize_flip always True
                )

            # ── Tensor initialisation ──────────────────────────────────────
            # hist_tensor: [MAX_AGENTS, T_H, 4 + num_extra]
            #   channels 0..1 : pos (dx,dy)
            #   channel  2    : timestamp
            #   channel  3    : existence mask
            #   channels 4..  : selected extra features
            hist_tensor = np.zeros((MAX_AGENTS, T_H, 4 + num_extra), dtype=np.float32)
            fut_tensor  = np.zeros((MAX_AGENTS, T_F, 3),             dtype=np.float32)
            pos_tensor  = np.zeros((MAX_AGENTS, 2),                   dtype=np.float32)

            # ── [Index 0] Ego ──────────────────────────────────────────────
            hist_tensor[0, :, 0:2] = ego_hist_rel
            hist_tensor[0, :, 2]   = timestamps
            hist_tensor[0, :, 3]   = 1.0
            if num_extra > 0:
                # Ego vs itself: dx=dy=0, compute full 13-ch feature and slice
                full_ego = np.zeros((T_H, FULL_DIM), np.float32)
                ego_class = vid_to_class.get(v, "Car")
                ego_phys_l = float(vid_to_w.get(v, 0.0))
                ego_phys_w = float(vid_to_h.get(v, 0.0))
                ego_size_bin, ego_volume = _volume_bin(ego_phys_l, ego_phys_w, ego_class)
                for ti in range(T_H):
                    lc_s, _lit_v, _lis_v, gate_v, ix_v, iy_v, i_v = compute_nb_features_scalar(
                        0.0, 0.0,
                        0.0, 0.0,
                        rot_ego_accs[ti, 0], rot_ego_accs[ti, 1],
                        nb_yv     = 0.0,
                        nb_lco    = 0.0,
                        nb_lco_norm = 0.0,
                        ki        = 0,
                        len_ego   = len_ego,
                        len_nb    = len_ego,
                        delta_lane = 0,
                        args      = args,
                        _lc_lane_lv  = _lc_lane_lv,
                        _lc_frame_ti = _lc_frame_ti,
                        _lc_type     = _lc_type,
                        _ti          = ti,
                    )
                    full_ego[ti] = [0.0, 0.0, 0.0, 0.0,
                                    rot_ego_accs[ti, 0], rot_ego_accs[ti, 1],
                                    lc_s, ego_volume, ego_size_bin, gate_v, ix_v, iy_v, i_v]
                hist_tensor[0, :, 4:] = full_ego[:, extra_indices]

            fut_tensor[0, :, :2] = ego_fut_rel
            fut_tensor[0, :, 2]  = 1.0
            pos_tensor[0]        = ego_hist_rel[-1]

            # ── [Index 1~8] Neighbors via 8-slot structure ─────────────────
            # We fill up to MAX_AGENTS-1 neighbors. We iterate over observed
            # neighbor slots at the obs timestep (t=T_H-1) and then fill in
            # other slots that appear in history.  Slot order determines agent
            # index, preserving the structured relationship used in neighformer.

            obs_row = ego_h_rows[-1]          # row for t=T_H-1
            ids8_obs = nb_ids_all[obs_row]    # 8 neighbor IDs at obs time

            # Collect all unique neighbor IDs that appear across history
            all_nb_ids: List[int] = []
            seen: set = set()
            # First, slot-ordered IDs at obs time (preserve slot structure)
            for ki in range(K):
                nid = int(ids8_obs[ki])
                if nid > 0 and nid not in seen:
                    all_nb_ids.append(nid)
                    seen.add(nid)
            # Then, fill from other hist timesteps (preserving appearance order)
            for ti, hr in enumerate(ego_h_rows[:-1]):
                for ki in range(K):
                    nid = int(nb_ids_all[hr, ki])
                    if nid > 0 and nid not in seen:
                        all_nb_ids.append(nid)
                        seen.add(nid)

            agent_count = 1
            for nid in all_nb_ids:
                if agent_count >= MAX_AGENTS:
                    break
                if nid == v:
                    continue
                rm = per_vid_frame_to_row.get(nid)
                if rm is None:
                    continue

                len_nb = float(vid_to_w.get(nid, 0.0))
                nb_class  = vid_to_class.get(nid, "Car")
                nb_phys_l = float(vid_to_w.get(nid, 0.0))   # CSV width = physical length
                nb_phys_w = float(vid_to_h.get(nid, 0.0))   # CSV height = physical width
                nb_size_bin, nb_volume = _volume_bin(nb_phys_l, nb_phys_w, nb_class)

                # Determine slot index ki for this neighbor (from obs-time slot)
                ki_nb = next((ki for ki in range(K) if int(ids8_obs[ki]) == nid), 0)

                # ── Full 13-ch feature array (T_H timesteps) ──────────────
                full_nb = np.zeros((T_H, FULL_DIM), np.float32)
                nb_mask_ti = np.zeros(T_H, bool)

                for ti, hf in enumerate(hist_frames):
                    r = rm.get(int(hf))
                    if r is None:
                        continue

                    # absolute neighbor state at this timestep
                    nb_x_abs = float(x[r]);   nb_y_abs = float(y[r])
                    nb_xv    = float(xv[r]);   nb_yv_raw = float(yv[r])
                    nb_xa    = float(xa[r]);   nb_ya    = float(ya[r])
                    nb_lco   = float(lat_lane_offset_arr[r])
                    nb_lw    = float(lat_lane_width_arr[r])
                    nb_lco_norm = nb_lco / (nb_lw * 0.5) if nb_lw > 0.5 else 0.0
                    nb_lane  = int(lane_id[r])

                    # ego state at this timestep
                    ego_x_abs = float(x[ego_h_rows[ti]])
                    ego_y_abs = float(y[ego_h_rows[ti]])

                    # relative features in rotated ego-centric frame
                    nb_pos_rot = transform_coord_vec(
                        np.array([[nb_x_abs, nb_y_abs]]), theta, norm_center
                    )[0]
                    nb_vel_rot = transform_coord_vec(
                        np.array([[nb_xv, nb_yv_raw]]), theta, np.zeros(2)
                    )[0]
                    nb_acc_rot = transform_coord_vec(
                        np.array([[nb_xa, nb_ya]]),  theta, np.zeros(2)
                    )[0]

                    dx_rot  = nb_pos_rot[0] - ego_hist_rel[ti, 0]
                    dy_rot  = nb_pos_rot[1] - ego_hist_rel[ti, 1]
                    dvx_rot = nb_vel_rot[0] - rot_ego_vels[ti, 0]
                    dvy_rot = nb_vel_rot[1] - rot_ego_vels[ti, 1]
                    ax_rot  = nb_acc_rot[0]
                    ay_rot  = nb_acc_rot[1]

                    delta_lane = abs(nb_lane - int(ego_lane_arr[ti]))

                    lc_s, _lit_v, _lis_v, gate_v, ix_v, iy_v, i_v = compute_nb_features_scalar(
                        dx_rot, dy_rot, dvx_rot, dvy_rot,
                        ax_rot, ay_rot,
                        nb_yv       = nb_yv_raw,
                        nb_lco      = nb_lco,
                        nb_lco_norm = nb_lco_norm,
                        ki          = ki_nb,
                        len_ego     = len_ego,
                        len_nb      = len_nb,
                        delta_lane  = delta_lane,
                        args        = args,
                        _lc_lane_lv  = _lc_lane_lv,
                        _lc_frame_ti = _lc_frame_ti,
                        _lc_type     = _lc_type,
                        _ti          = ti,
                    )

                    full_nb[ti] = [dx_rot, dy_rot, dvx_rot, dvy_rot,
                                   ax_rot, ay_rot,
                                   lc_s, nb_volume, nb_size_bin, gate_v, ix_v, iy_v, i_v]
                    nb_mask_ti[ti] = True

                if not nb_mask_ti.any():
                    continue

                # ── Top-N gate (per timestep) ────────────────────────────
                if args.gate_topn > 0:
                    # Build a single-slot view for top-N (we have one slot per agent here)
                    # Full top-N is only meaningful when processing all slots at once;
                    # here we approximate by applying gate_theta only (top-N in neighformer
                    # operates across K slots simultaneously, which is meaningful in the
                    # slot-structured tensor).  For mmT's agent-ordered layout, we apply
                    # gate_theta as the effective filter.
                    pass  # gate already set via gate_theta in compute_nb_features_scalar

                # ── gate_mask: if gate=0 at ALL timesteps, skip agent ────
                if args.gate_mask:
                    if not np.any(full_nb[:, 9] > 0):
                        continue

                # ── Fill hist_tensor ──────────────────────────────────────
                hist_tensor[agent_count, :, 0:2] = full_nb[:, 0:2]
                hist_tensor[agent_count, :, 2]   = timestamps
                hist_tensor[agent_count, :, 3]   = nb_mask_ti.astype(np.float32)
                if num_extra > 0:
                    hist_tensor[agent_count, :, 4:] = full_nb[:, extra_indices]

                # pos at obs time (last hist step)
                if nb_mask_ti[T_H - 1]:
                    pos_tensor[agent_count] = full_nb[T_H - 1, 0:2]

                # ── Fill fut_tensor ───────────────────────────────────────
                for fi, ff in enumerate(fut_frames):
                    r = rm.get(int(ff))
                    if r is None:
                        continue
                    nb_fut_rot = transform_coord_vec(
                        np.array([[float(x[r]), float(y[r])]]), theta, norm_center
                    )[0]
                    fut_tensor[agent_count, fi, :2] = nb_fut_rot
                    fut_tensor[agent_count, fi, 2]  = 1.0

                agent_count += 1

            # ── Save sample ──────────────────────────────────────────────
            lane_ids_list   = list(lane_id2idx.values())[:MAX_LANES]
            valid_lane_num  = len(lane_ids_list)
            padded_lane_ids = lane_ids_list + [-1] * (MAX_LANES - valid_lane_num)

            out_name.append(f"{rec_id}_{v}_{obs_frame_val}")
            out_meta_rec.append(int(rec_id))
            out_meta_track.append(int(v))
            out_meta_frame.append(int(obs_frame_val))
            out_city.append(city_name)
            out_hist.append(hist_tensor)
            out_fut.append(fut_tensor)
            out_lane_id.append(padded_lane_ids)
            out_norm.append(norm_center)
            out_theta.append(theta)
            out_pos.append(pos_tensor)
            out_valid.append([agent_count, valid_lane_num])

            obs_frame_val += slide_step * ds_stride

    # ── Write temp HDF5 ──────────────────────────────────────────────────────
    if out_hist:
        temp_file = temp_dir / f"{rec_id}.h5"
        dt_str = h5py.string_dtype(encoding='utf-8')
        with h5py.File(temp_file, 'w') as f:
            f.create_dataset('NAME',       data=np.array(out_name, dtype=object), dtype=dt_str)
            f.create_dataset('CITY_NAME',  data=np.array(out_city, dtype=object), dtype=dt_str)
            f.create_dataset('HISTORY',    data=np.array(out_hist,     np.float32), compression="gzip")
            f.create_dataset('FUTURE',     data=np.array(out_fut,      np.float32), compression="gzip")
            f.create_dataset('LANE_ID',    data=np.array(out_lane_id,  np.int32),   compression="gzip")
            f.create_dataset('NORM_CENTER',data=np.array(out_norm,     np.float32), compression="gzip")
            f.create_dataset('THETA',      data=np.array(out_theta,    np.float32), compression="gzip")
            f.create_dataset('POS',        data=np.array(out_pos,      np.float32), compression="gzip")
            f.create_dataset('VALID_LEN',  data=np.array(out_valid,    np.int32),   compression="gzip")
            f.create_dataset('META_REC',   data=np.array(out_meta_rec,   np.int32))
            f.create_dataset('META_TRACK', data=np.array(out_meta_track, np.int32))
            f.create_dataset('META_FRAME', data=np.array(out_meta_frame, np.int32))

    return rec_id, map_dict, global_lane_id2idx


# ==============================================================================
# 9. Multiprocessing wrapper
# ==============================================================================
def process_recording_wrapper(args_tuple):
    try:
        return process_recording(*args_tuple)
    except Exception as e:
        print(f"\n[Error in Rec {args_tuple[0]}]: {e}")
        traceback.print_exc()
        return None, {}, {}


# ==============================================================================
# 10. Train / Val / Test split
# ==============================================================================
def balanced_recording_split(ds_counts: dict, ratios=(0.7, 0.1, 0.2), seed=42):
    rng = np.random.default_rng(seed)
    total_samples = sum(ds_counts.values())
    targets = [total_samples * r for r in ratios]
    items   = list(ds_counts.items())
    rng.shuffle(items)
    items.sort(key=lambda x: x[1], reverse=True)

    splits = {"train": [], "val": [], "test": []}
    sums   = {"train": 0,  "val": 0,  "test": 0}
    keys   = ["train", "val", "test"]

    for rec_id, cnt in items:
        deficits = {k: (targets[j] - sums[k]) for j, k in enumerate(keys)}
        best = max(deficits.items(), key=lambda kv: kv[1])[0]
        splits[best].append(rec_id)
        sums[best] += cnt

    return splits, sums


# ==============================================================================
# 11. HDF5 merge
# ==============================================================================
def merge_h5_files(file_list, out_file):
    if not file_list:
        return

    total_rows = 0
    shapes, dtypes = {}, {}

    with h5py.File(file_list[0], 'r') as f_sample:
        for k in f_sample.keys():
            shapes[k] = f_sample[k].shape[1:]
            dtypes[k] = f_sample[k].dtype

    for f_path in file_list:
        with h5py.File(f_path, 'r') as f:
            total_rows += f['HISTORY'].shape[0]

    if total_rows == 0:
        return

    with h5py.File(out_file, 'w') as out_f:
        dsets = {}
        for k in shapes.keys():
            if dtypes[k].kind == 'O':
                dsets[k] = out_f.create_dataset(k, shape=(total_rows,) + shapes[k], dtype=dtypes[k])
            else:
                dsets[k] = out_f.create_dataset(k, shape=(total_rows,) + shapes[k],
                                                 dtype=dtypes[k], compression="gzip")
        current_idx = 0
        for f_path in file_list:
            with h5py.File(f_path, 'r') as in_f:
                n = in_f['HISTORY'].shape[0]
                if n == 0:
                    continue
                for k in shapes.keys():
                    dsets[k][current_idx: current_idx + n] = in_f[k][:]
                current_idx += n


# ==============================================================================
# 12. Entry point
# ==============================================================================
def main():
    args = parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir) / args.feature_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_ids = sorted(set([
        re.match(r"(\d+)_tracks\.csv$", p.name).group(1)
        for p in raw_dir.glob("*_tracks.csv")
        if re.match(r"(\d+)_tracks\.csv$", p.name)
    ]))

    temp_dir = out_dir / "temp_records"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] feature_mode={args.feature_mode}  lc_version={args.lc_version}  "
          f"importance_mode={args.importance_mode}  lis_mode={args.lis_mode}")
    print(f"         gate_theta={args.gate_theta}  gate_topn={args.gate_topn}  "
          f"gate_mask={args.gate_mask}  slot_alpha={args.slot_importance_alpha}"
          + ("  slot_conditional=True" if args.slot_importance_conditional else ""))
    print(f"         eps_gate={args.eps_gate}  t_front={args.t_front}  t_back={args.t_back}")
    print(f"[Output] {out_dir}")
    print(f"Starting multiprocessing with {os.cpu_count()} cores ...")

    final_map_dict    = {}
    final_lane_id2idx = {}

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(
            executor.map(
                process_recording_wrapper,
                [(rec_id, raw_dir, temp_dir, args) for rec_id in rec_ids],
                chunksize=1,
            ),
            total=len(rec_ids), desc="Processing HighD",
        ))
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
    print(f"Split (sample counts): Train={sums['train']}  Val={sums['val']}  Test={sums['test']}")

    for split_name, split_rec_ids in splits.items():
        if not split_rec_ids:
            continue
        print(f"Merging {split_name} set ...")
        file_list = [temp_dir / f"{rec_id}.h5" for rec_id in split_rec_ids]
        out_file  = out_dir / f"{split_name}.h5"
        merge_h5_files(file_list, out_file)
        print(f"  -> {out_file}")

    with open(out_dir / "map.pkl", "wb") as f:
        pickle.dump({"Map": final_map_dict, "Lane_id2idx": final_lane_id2idx}, f)
        print(f"  -> {out_dir / 'map.pkl'}")

    print("All done!")


if __name__ == "__main__":
    main()