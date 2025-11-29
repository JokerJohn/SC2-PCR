import argparse
import json
import logging
import os
import os.path as osp
import sys
import threading
import time
from typing import List, Optional, Tuple

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from SC2_PCR import Matcher


# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
WORKING_DIR = osp.dirname(osp.realpath(__file__))
ROOT_DIR = WORKING_DIR

# SC2-PCR 配准相关固定参数（独立于 custom_infer）
KITTI_CONFIG_PATH = osp.join(ROOT_DIR, 'config_json', 'config_KITTI.json')
INLIER_THRESHOLD = 0.2  # 米，用于评估
MAX_MATCHER_ITERATIONS = 30
NORMAL_KNN = 5
SEARCH_RADIUS_METERS = 1.0


# 固定路径配置：将数据放在同一文件夹内，仅需修改这里
# DATA_DIR = '/home/xchu/data/ltloc_result/geo_transformer/parkinglot_raw_geo'  # TODO: 修改为包含先验地图、帧点云和 TUM 位姿的目录
# PRIOR_MAP_FILENAME = 'parkinglot_raw.pcd'

# 固定路径配置：将数据放在同一文件夹内，仅需修改这里
# DATA_DIR = '/home/xchu/data/ltloc_result/geo_transformer/stairs_bob_geo'
# PRIOR_MAP_FILENAME = 'stairs_bob.pcd'

# DATA_DIR='/home/xchu/data/ltloc_result/geo_transformer/20220216_corridor_day_ref_geo'
# PRIOR_MAP_FILENAME = '20220216_corridor_day_ref.pcd'

# DATA_DIR = '/home/xchu/data/ltloc_result/geo_transformer/20220225_building_day_ref_geo'
# PRIOR_MAP_FILENAME = '20220225_building_day_ref.pcd'

DATA_DIR = '/home/xchu/data/ltloc_result/geo_transformer/cave02_geo'
PRIOR_MAP_FILENAME = 'lauren_cavern01.pcd'


TUM_FILENAME = 'optimized_poses_tum.txt'
OUTPUT_TUM_FILENAME_FPFH = 'sc2pcr_fpfh.txt'
OUTPUT_TUM_FILENAME_RAW = 'sc2pcr_raw.txt'
FRAME_METRICS_FILENAME = 'frame_metrics.txt'
SUMMARY_METRICS_FILENAME = 'metrics_summary.txt'
RUNTIME_PROFILE_FILENAME = 'sc2pcr_runtime.txt'
RUNTIME_SUMMARY_FILENAME = 'runtime_profile_summary.txt'
MERGED_MAP_FILENAME = 'sc2pcr_merged_map.pcd'
RESULT_ROOT_DIR = 'sc2-pcr_results'  # 顶层输出目录，避免覆盖其它方法
RESULT_SUBDIR_FPFH = 'fpfh'
RESULT_SUBDIR_RAW = 'raw'
KEYFRAME_SUBDIR = 'key_point_frame'

# 可选：将 LIO 里程计坐标系统一变换到地图坐标系的初始位姿
# 2) 使用 RPY 形式：将 INITIAL_TRANSFORM 设为 None，
#    再把 INITIAL_POSE 填成 (tx, ty, tz, roll, pitch, yaw)，角度单位由 POSE_IN_RADIANS 决定

# INITIAL_TRANSFORM =  [ 0.556912342, -0.830538784, -0.007341485,-223.941216745,  #PK01
#                   0.830297811 ,0.556480475 , 0.030576983 ,-534.004804865,
#                   -0.021309977,-0.023124317 , 0.999505452, -1.854225961,
#                   0.000000000 , 0.000000000, 0.000000000 , 1.000000000 ]
# INITIAL_TRANSFORM =  [ -0.519301, 0.850557, 0.082936 ,-11.347226, #stairs
#                   -0.852164, -0.522691, 0.024698, 3.002144,
#                   0.064357, -0.057849, 0.996249, -0.715776,
#                   0.000000, 0.000000, 0.000000, 1.000000 ]
# INITIAL_TRANSFORM = [0.962393,  -0.269109 , 0.0371485 ,   6.26396, #CORRIDOR
#                   0.267793,   0.962772 , 0.0368319  ,0.0850816,
#                   -0.0456773, -0.0254987  , 0.998631  , 0.792745,
#                   0,        0,       0,        1 ]
# building day
# INITIAL_TRANSFORM = [ 0.448165, -0.893951 ,-0.000477, -41.163830,
#                   0.891230, 0.446760 ,0.078197, -46.008873,
#                   -0.069691, -0.035470, 0.996938, 0.261990,
#                   0.000000, 0.000000, 0.000000, 1.000000 ]

# cave02
INITIAL_TRANSFORM = [ 0.274473488331, 0.961305737495, 0.023577133194,33.809207916260,
                  -0.961209475994, 0.274974942207, -0.021568179131,-72.459877014160,
                  -0.027216726914, -0.016742672771, 0.999489367008,21.207557678223,
                  0,      0,     0,      1 ]


INITIAL_POSE = None
POSE_IN_RADIANS = False

def load_kitti_config(path: str) -> dict:
    if not osp.isfile(path):
        raise FileNotFoundError(f'KITTI 配置文件不存在: {path}')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_matcher(config: dict) -> Matcher:
    return Matcher(
        inlier_threshold=config['inlier_threshold'],
        num_node=config['num_node'],
        use_mutual=config['use_mutual'],
        d_thre=config['d_thre'],
        num_iterations=config['num_iterations'],
        ratio=config['ratio'],
        nms_radius=config['nms_radius'],
        max_points=config['max_points'],
        k1=config['k1'],
        k2=config['k2'],
    )


def get_descriptor_settings(config: dict) -> dict:
    return {
        'type': config.get('descriptor', 'fpfh').lower(),
        'normal_knn': NORMAL_KNN,
        'feature_radius': SEARCH_RADIUS_METERS,
        'feature_max_nn': 100,
    }


def compute_fpfh_features(points: np.ndarray, descriptor_cfg: dict):
    import open3d as o3d

    if points.shape[0] == 0:
        raise ValueError('点云为空，无法计算 FPFH 特征。')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if hasattr(o3d, 'pipelines') and hasattr(o3d.pipelines, 'registration'):
        registration_mod = o3d.pipelines.registration
    elif hasattr(o3d, 'registration'):
        registration_mod = o3d.registration
    else:
        raise AttributeError('当前 Open3D 版本缺少 registration 模块，无法计算 FPFH 特征。')

    normal_knn = descriptor_cfg.get('normal_knn', NORMAL_KNN)
    feature_radius = descriptor_cfg.get('feature_radius', SEARCH_RADIUS_METERS)
    feature_max_nn = descriptor_cfg.get('feature_max_nn', 100)
    normal_param = o3d.geometry.KDTreeSearchParamKNN(normal_knn)
    feature_param = o3d.geometry.KDTreeSearchParamHybrid(radius=feature_radius, max_nn=feature_max_nn)
    pcd.estimate_normals(normal_param)
    fpfh = registration_mod.compute_fpfh_feature(pcd, feature_param)
    points_ds = np.asarray(pcd.points, dtype=np.float32)
    features = np.asarray(fpfh.data, dtype=np.float32).T
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)
    return points_ds, features


def build_raw_point_features(points: np.ndarray):
    keypoints = points.astype(np.float32)
    centered = keypoints - keypoints.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    features = centered / norms
    return keypoints, features


def prepare_matcher_inputs(points: np.ndarray, device: torch.device, descriptor_cfg=None, use_fpfh=False):
    if points.shape[0] == 0:
        raise ValueError('点云为空，无法构建匹配输入。')
    if use_fpfh:
        if descriptor_cfg is None:
            raise ValueError('使用 FPFH 流程时必须提供 descriptor 配置。')
        keypoints, features = compute_fpfh_features(points, descriptor_cfg)
    else:
        keypoints, features = build_raw_point_features(points)
    keypoints_tensor = torch.from_numpy(keypoints).float().to(device)[None, ...]
    features_tensor = torch.from_numpy(features).float().to(device)[None, ...]
    return keypoints_tensor, features_tensor


def load_point_cloud(path: str) -> np.ndarray:
    ext = osp.splitext(path)[1].lower()
    if ext == '.npy':
        points = np.load(path)
    elif ext in ['.pcd', '.ply', '.xyz']:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) == 0:
            raise ValueError(f'Point cloud {path} is empty.')
        points = np.asarray(pcd.points)
    elif ext == '.bin':
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')
    return points.astype(np.float32)


def get_transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform.astype(np.float32)


def pose_to_transform(pose, radians: bool = False) -> np.ndarray:
    """将 (tx, ty, tz, r, p, y) 转为 4x4 齐次变换矩阵。"""
    if pose is None:
        raise ValueError('INITIAL_POSE 不能为空。')
    translation = np.asarray(pose[:3], dtype=np.float64)
    angles = np.asarray(pose[3:], dtype=np.float64)
    rotation = Rotation.from_euler('xyz', angles, degrees=not radians).as_matrix()
    transform = get_transform_from_rotation_translation(rotation, translation)
    return transform.astype(np.float64)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogeneous = np.concatenate([points, ones], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def _knn_query(tree, points):
    for kwargs in ({'workers': -1}, {'n_jobs': -1}, {}):
        try:
            return tree.query(points, k=1, **kwargs)
        except TypeError:
            continue
    return tree.query(points, k=1)


def compute_pointwise_metrics(aligned_points, target_points, inlier_threshold):
    if aligned_points.shape[0] == 0 or target_points.shape[0] == 0:
        return np.nan, np.nan, 0.0, 0

    target_tree = cKDTree(target_points)
    dists_src_to_tgt, _ = _knn_query(target_tree, aligned_points)

    source_tree = cKDTree(aligned_points)
    dists_tgt_to_src, _ = _knn_query(source_tree, target_points)

    chamfer = float(dists_src_to_tgt.mean() + dists_tgt_to_src.mean())

    inlier_mask = dists_src_to_tgt <= inlier_threshold
    inlier_count = int(inlier_mask.sum())
    if inlier_count > 0:
        rmse = float(np.sqrt(np.mean(np.square(dists_src_to_tgt[inlier_mask]))))
    else:
        rmse = float('inf')
    fitness = inlier_count / aligned_points.shape[0]

    return chamfer, rmse, fitness, inlier_count


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if points.shape[0] == 0 or voxel_size <= 0:
        return points
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downsampled.points, dtype=np.float32)


def parse_tum_file(path: str) -> List[dict]:
    poses = []
    with open(path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                logging.warning('Line %d in %s is invalid: %s', line_idx, path, line)
                continue
            timestamp = float(parts[0])
            translation = np.asarray(parts[1:4], dtype=np.float64)
            quat = np.asarray(parts[4:8], dtype=np.float64)
            norm = np.linalg.norm(quat)
            if norm == 0:
                logging.warning('Quaternion on line %d is zero norm, skip.', line_idx)
                continue
            quat = quat / norm
            rotation = Rotation.from_quat(quat).as_matrix()
            transform = get_transform_from_rotation_translation(rotation, translation)
            poses.append({
                'timestamp': timestamp,
                'transform': transform.astype(np.float32),
            })
    return poses


def crop_prior_map(map_points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """在 XY 平面上按半径裁剪局部地图，仿照 PCL CropBox 的思路实现。"""
    if map_points.shape[0] == 0:
        logging.info('crop_prior_map: 输入点云为空')
        return map_points

    cx, cy = float(center[0]), float(center[1])
    r2 = float(radius) * float(radius)
    xy = map_points[:, :2]
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist2 = dx * dx + dy * dy
    mask = dist2 <= r2
    cropped = map_points[mask]
    logging.info('crop_prior_map: 半径=%.2f, 输入点数=%d, 输出点数=%d', radius, map_points.shape[0], cropped.shape[0])
    return cropped


def run_single_registration(
    matcher,
    device: torch.device,
    ref_points: np.ndarray,
    src_points: np.ndarray,
    descriptor_cfg: Optional[dict],
    use_fpfh: bool,
    inlier_threshold: float,
):
    if ref_points.shape[0] == 0 or src_points.shape[0] == 0:
        raise ValueError('Empty point cloud encountered during registration.')

    if device.type == 'cuda':
        torch.cuda.synchronize()
    prep_start = time.perf_counter()
    src_keypts, src_feats = prepare_matcher_inputs(src_points, device, descriptor_cfg, use_fpfh)
    ref_keypts, ref_feats = prepare_matcher_inputs(ref_points, device, descriptor_cfg, use_fpfh)
    prep_time_ms = (time.perf_counter() - prep_start) * 1000.0

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    with torch.no_grad():
        est_transform, _, _, _ = matcher.estimator(src_keypts, ref_keypts, src_feats, ref_feats)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    est_transform_np = est_transform.squeeze(0).detach().cpu().numpy().astype(np.float64)
    aligned_src_points = apply_transform(src_points.copy(), est_transform_np.astype(np.float32))
    chamfer, rmse, fitness, inlier_count = compute_pointwise_metrics(
        aligned_src_points, ref_points, inlier_threshold
    )
    return est_transform_np, runtime_ms, prep_time_ms, chamfer, rmse, fitness, inlier_count


def write_tum_file(path: str, results: List[Tuple[float, np.ndarray]], append: bool = False):
    mode = 'a' if append else 'w'
    with open(path, mode) as f:
        for timestamp, transform in results:
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            quat = Rotation.from_matrix(rotation).as_quat()
            line = [
                f'{timestamp:.9f}',
                f'{translation[0]:.8f}',
                f'{translation[1]:.8f}',
                f'{translation[2]:.8f}',
                f'{quat[0]:.8f}',
                f'{quat[1]:.8f}',
                f'{quat[2]:.8f}',
                f'{quat[3]:.8f}',
            ]
            f.write(' '.join(line) + '\n')


def write_frame_metrics_txt(path: str, logs: List[dict]):
    header = '# index timestamp scan_path runtime_ms chamfer rmse fitness inlier_count correction_matrix(16 values row-major)\n'
    with open(path, 'w') as f:
        f.write(header)
        for log in logs:
            matrix_str = ' '.join(f'{v:.8f}' for v in log['matrix'])
            line = (
                f"{log['index']} "
                f"{log['timestamp']:.9f} "
                f"{log['scan_path']} "
                f"{log['runtime_ms']:.3f} "
                f"{log['chamfer']:.6f} "
                f"{log['rmse']:.6f} "
                f"{log['fitness']:.6f} "
                f"{log['inlier_count']} "
                f"{matrix_str}"
            )
            f.write(line + '\n')


def write_summary_metrics_txt(path: str, logs: List[dict]):
    if not logs:
        return

    def avg(key):
        values = [log[key] for log in logs if np.isfinite(log[key])]
        if not values:
            return float('nan')
        return float(np.mean(values))

    metrics = {
        'frame_count': len(logs),
        'avg_runtime_ms': avg('runtime_ms'),
        'avg_chamfer': avg('chamfer'),
        'avg_rmse': avg('rmse'),
        'avg_fitness': avg('fitness'),
        'avg_inlier_count': avg('inlier_count'),
    }

    with open(path, 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f'{key}: {value:.6f}\n')
            else:
                f.write(f'{key}: {value}\n')


def init_runtime_profile_txt(path: str):
    header = '# index timestamp ref_points src_points crop_time_ms prep_time_ms registration_time_ms total_time_ms\n'
    with open(path, 'w') as f:
        f.write(header)


def append_runtime_profile_txt(path: str, log: dict):
    line = (
        f"{log['index']} "
        f"{log['timestamp']:.9f} "
        f"{log['ref_points']} "
        f"{log['src_points']} "
        f"{log['crop_time_ms']:.3f} "
        f"{log['prep_time_ms']:.3f} "
        f"{log['registration_time_ms']:.3f} "
        f"{log['total_time_ms']:.3f}\n"
    )
    with open(path, 'a') as f:
        f.write(line)


def write_runtime_summary_txt(path: str, logs: List[dict]):
    keys = ['crop_time_ms', 'prep_time_ms', 'registration_time_ms', 'total_time_ms']
    with open(path, 'w') as f:
        if not logs:
            f.write('# 没有可用的帧耗时统计\n')
            return
        for key in keys:
            values = [log[key] for log in logs]
            avg_value = float(np.mean(values))
            f.write(f'avg_{key}: {avg_value:.6f}\n')


def flush_merged_map(merged_points_list: List[np.ndarray], output_path: str) -> bool:
    if not output_path:
        return False
    if not merged_points_list:
        logging.warning('当前没有可用于融合的点云，无法写出融合地图。')
        return False
    merged_points = np.concatenate(merged_points_list, axis=0)
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        o3d.io.write_point_cloud(output_path, pcd)
        logging.info('融合地图已保存到 %s，总点数 %d', output_path, merged_points.shape[0])
        return True
    except Exception as err:
        logging.warning('写入融合地图时出错：%s', err)
        return False


def start_control_listener(stop_event: threading.Event, merge_event: threading.Event):
    if not sys.stdin.isatty():
        logging.info('标准输入非交互，按键控制不可用。')
        return None

    def _listener():
        logging.info('控制提示：按一次空格即可请求停止，按 m 可立即输出融合地图。')
        fd = sys.stdin.fileno()
        use_raw = termios is not None and tty is not None
        old_settings = None
        if use_raw:
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        try:
            while not stop_event.is_set():
                if use_raw:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    if ch == ' ':
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    lowered = ch.lower()
                    if lowered == 'm':
                        logging.info('收到立即融合指令，将尽快写出融合地图。')
                        merge_event.set()
                else:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    stripped = line.strip().lower()
                    if line.strip() == '' and ' ' in line:
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    if stripped in {'space', 'stop'}:
                        logging.info('收到停止指令，将在当前帧结束后停止处理。')
                        stop_event.set()
                        break
                    if stripped == 'm':
                        logging.info('收到立即融合指令，将尽快写出融合地图。')
                        merge_event.set()
        finally:
            if use_raw and old_settings is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    thread = threading.Thread(target=_listener, daemon=True)
    thread.start()
    return thread


def parse_args():
    parser = argparse.ArgumentParser(description='使用 SC2-PCR 配准器对序列数据进行逐帧配准并输出 TUM 轨迹。')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR, help='包含先验地图、点云帧和 TUM 初值的目录 (默认使用上方常量)')
    parser.add_argument('--scan_dir', type=str, default=None, help='若不设置则默认为 data_dir/key_point_frame')
    parser.add_argument('--map_path', type=str, default=None, help='若不设置则默认为 data_dir + PRIOR_MAP_FILENAME')
    parser.add_argument('--tum_path', type=str, default=None, help='若不设置则默认为 data_dir + TUM_FILENAME')
    parser.add_argument('--output_tum', type=str, default=None, help='若不设置则根据是否使用 FPFH 自动命名')
    parser.add_argument('--scan_extension', type=str, default='.pcd', help='单帧点云的扩展名 (默认 .pcd)')
    parser.add_argument('--scan_voxel_size', type=float, default=0.2, help='单帧体素下采样分辨率 (米)')
    parser.add_argument('--map_voxel_size', type=float, default=0.5, help='先验地图体素下采样分辨率 (米)')
    parser.add_argument('--crop_radius', type=float, default=100.0, help='裁剪先验地图的半径 (米)')
    parser.add_argument('--use_fpfh', dest='use_fpfh', action='store_false', default=False, help='使用 FPFH 特征进行配准（默认开启）')
    parser.add_argument('--no_fpfh', dest='use_fpfh', action='store_true', help='关闭 FPFH，改用原始点特征')
    parser.add_argument('--config', type=str, default=KITTI_CONFIG_PATH, help='SC2-PCR 配置文件路径 (默认 config_json/config_KITTI.json)')
    parser.add_argument('--matcher_iterations', type=int, default=MAX_MATCHER_ITERATIONS, help='Matcher 迭代次数')
    parser.add_argument('--inlier_threshold', type=float, default=INLIER_THRESHOLD, help='评估内点阈值 (米)')
    parser.add_argument('--normal_knn', type=int, default=NORMAL_KNN, help='FPFH 法向估计 KNN')
    parser.add_argument('--feature_radius', type=float, default=SEARCH_RADIUS_METERS, help='FPFH 邻域半径 (米)')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--log_path', type=str, default=None, help='若提供则保存 JSON 日志')
    parser.add_argument('--frame_metrics', type=str, default=None, help='逐帧指标 txt（默认 data_dir/frame_metrics.txt）')
    parser.add_argument('--summary_metrics', type=str, default=None, help='平均指标 txt（默认 data_dir/metrics_summary.txt）')
    parser.add_argument('--runtime_profile', type=str, default=None, help='逐帧耗时统计 txt（默认 data_dir/runtime_profile.txt）')
    parser.add_argument('--runtime_summary', type=str, default=None, help='耗时汇总 txt（默认 data_dir/runtime_profile_summary.txt）')
    parser.add_argument('--merged_map', type=str, default=None, help='融合后的全局地图 pcd 路径（默认 data_dir/merged_map.pcd）')
    # 需求：参数固定为常量，不从命令行接收，因此显式传空列表。
    return parser.parse_args(args=[])


def run_mode(
    args,
    use_fpfh: bool,
    prior_map: np.ndarray,
    poses: List[dict],
    global_init_transform: Optional[np.ndarray],
):
    base_result_dir = osp.join(args.data_dir, RESULT_ROOT_DIR)
    method_subdir = RESULT_SUBDIR_FPFH if use_fpfh else RESULT_SUBDIR_RAW
    result_dir = osp.join(base_result_dir, method_subdir)
    os.makedirs(result_dir, exist_ok=True)

    output_tum = args.output_tum or osp.join(result_dir, OUTPUT_TUM_FILENAME_FPFH if use_fpfh else OUTPUT_TUM_FILENAME_RAW)
    frame_metrics = args.frame_metrics or osp.join(result_dir, FRAME_METRICS_FILENAME)
    summary_metrics = args.summary_metrics or osp.join(result_dir, SUMMARY_METRICS_FILENAME)
    runtime_profile = args.runtime_profile or osp.join(result_dir, RUNTIME_PROFILE_FILENAME)
    runtime_summary = args.runtime_summary or osp.join(result_dir, RUNTIME_SUMMARY_FILENAME)
    merged_map_path = args.merged_map or osp.join(result_dir, MERGED_MAP_FILENAME)

    logging.info('结果文件将输出到子目录：%s', result_dir)
    logging.info('配准流程与 custom_infer 完全一致，use_fpfh=%s', use_fpfh)

    config = load_kitti_config(args.config)
    config = dict(config)
    config['num_iterations'] = args.matcher_iterations
    config['inlier_threshold'] = args.inlier_threshold

    descriptor_cfg = None
    if use_fpfh:
        descriptor_cfg = get_descriptor_settings(config)
        descriptor_cfg['normal_knn'] = args.normal_knn
        descriptor_cfg['feature_radius'] = args.feature_radius
    matcher = build_matcher(config)

    if torch.cuda.is_available() and not args.force_cpu:
        device = torch.device('cuda')
        logging.info('使用 CUDA 进行推理')
    else:
        device = torch.device('cpu')
        if args.force_cpu:
            logging.info('已按照参数设置强制使用 CPU')
        else:
            logging.info('CUDA 不可用，使用 CPU')

    stop_event = threading.Event()
    merge_event = threading.Event()
    start_control_listener(stop_event, merge_event)

    refined_results = []
    per_frame_logs = []
    runtime_profile_logs = []
    merged_points_list = []

    write_tum_file(output_tum, [], append=False)
    logging.info('清空/创建输出 TUM 轨迹文件：%s', output_tum)
    init_runtime_profile_txt(runtime_profile)
    logging.info('清空/创建逐帧耗时文件：%s', runtime_profile)

    for idx, pose in enumerate(poses):
        if stop_event.is_set():
            logging.info('检测到停止指令，提前结束后续帧处理。')
            break
        scan_path = osp.join(args.scan_dir, f'{idx}{args.scan_extension}')
        if not osp.isfile(scan_path):
            logging.warning('点云 %s 不存在，跳过。', scan_path)
            continue

        scan_points = load_point_cloud(scan_path)
        logging.info('帧 %d 原始点数：%d', idx, scan_points.shape[0])
        scan_points = voxel_downsample(scan_points, args.scan_voxel_size)
        logging.info('帧 %d 体素下采样 (%.3fm) 后点数：%d', idx, args.scan_voxel_size, scan_points.shape[0])
        if scan_points.shape[0] == 0:
            logging.warning('点云 %s 在下采样后为空，跳过。', scan_path)
            continue

        pose_transform = pose['transform'].astype(np.float64)
        if global_init_transform is not None:
            init_transform = global_init_transform @ pose_transform
        else:
            init_transform = pose_transform
        if idx == 0:
            logging.info('首帧 init_transform:\n%s', init_transform)
        scan_in_map = apply_transform(scan_points.copy(), init_transform.astype(np.float32))

        center = init_transform[:3, 3]
        logging.info('帧 %d 裁剪先验地图，中心=(%.3f, %.3f, %.3f), 半径=%.2fm', idx, center[0], center[1], center[2], args.crop_radius)
        crop_start = time.perf_counter()
        cropped_map = crop_prior_map(prior_map, center, args.crop_radius)
        crop_time_ms = (time.perf_counter() - crop_start) * 1000.0
        logging.info('帧 %d 裁剪耗时：%.2f ms', idx, crop_time_ms)
        if cropped_map.shape[0] == 0:
            logging.warning('基于初始位姿裁剪先验地图为空，使用完整先验地图。')
            cropped_map = prior_map
        ref_points_count = int(cropped_map.shape[0])
        src_points_count = int(scan_points.shape[0])

        try:
            correction, registration_time_ms, prep_time_ms, chamfer, rmse, fitness, inlier_count = run_single_registration(
                matcher,
                device,
                cropped_map,
                scan_in_map,
                descriptor_cfg,
                use_fpfh,
                args.inlier_threshold,
            )
        except ValueError as err:
            logging.warning('注册帧 %d 失败：%s', idx, err)
            continue

        refined_transform = (correction @ init_transform).astype(np.float64)
        refined_results.append((pose['timestamp'], refined_transform))
        write_tum_file(output_tum, [(pose['timestamp'], refined_transform)], append=True)
        logging.info('帧 %d 的优化姿态已写入 TUM 文件。', idx)
        if merged_map_path:
            registered_points = apply_transform(scan_points.copy(), refined_transform.astype(np.float32))
            merged_points_list.append(registered_points.astype(np.float32))
            logging.info('帧 %d 点云已加入融合地图，点数 %d', idx, registered_points.shape[0])

        total_time_ms = crop_time_ms + prep_time_ms + registration_time_ms
        runtime_profile_logs.append({
            'index': idx,
            'timestamp': pose['timestamp'],
            'ref_points': ref_points_count,
            'src_points': src_points_count,
            'crop_time_ms': crop_time_ms,
            'prep_time_ms': prep_time_ms,
            'registration_time_ms': registration_time_ms,
            'total_time_ms': total_time_ms,
        })
        append_runtime_profile_txt(runtime_profile, runtime_profile_logs[-1])

        per_frame_logs.append({
            'index': idx,
            'scan_path': scan_path,
            'timestamp': pose['timestamp'],
            'runtime_ms': registration_time_ms,
            'chamfer': chamfer,
            'rmse': rmse,
            'fitness': fitness,
            'inlier_count': inlier_count,
            'matrix': correction.astype(np.float64).reshape(-1).tolist(),
        })
        logging.info(
            '帧 %d 完成：Chamfer %.4f，RMSE %.4f，Fitness %.4f，Registration %.1f ms，点云 %s',
            idx,
            chamfer,
            rmse,
            fitness,
            registration_time_ms,
            scan_path,
        )
        logging.info(
            '帧 %d 耗时统计：裁剪 %.2f ms + 预处理 %.2f ms + 注册 %.2f ms = %.2f ms',
            idx,
            crop_time_ms,
            prep_time_ms,
            registration_time_ms,
            total_time_ms,
        )
        if merged_map_path and merge_event.is_set():
            flush_merged_map(merged_points_list, merged_map_path)
            merge_event.clear()
        if stop_event.is_set():
            logging.info('收到停止指令，在处理完帧 %d 后退出。', idx)
            break

    if not refined_results:
        if stop_event.is_set():
            logging.warning('由于用户请求停止，未成功注册任何帧。')
            return
        raise RuntimeError('未成功注册任何帧。')

    logging.info('已实时写入全部优化轨迹到 %s', output_tum)

    write_frame_metrics_txt(frame_metrics, per_frame_logs)
    logging.info('逐帧指标已保存到 %s', frame_metrics)

    write_summary_metrics_txt(summary_metrics, per_frame_logs)
    logging.info('平均指标已保存到 %s', summary_metrics)

    write_runtime_summary_txt(runtime_summary, runtime_profile_logs)
    logging.info('逐帧耗时统计汇总已保存到 %s', runtime_summary)

    if merged_map_path:
        flush_merged_map(merged_points_list, merged_map_path)

    if args.log_path is not None:
        with open(args.log_path, 'w') as f:
            json.dump(per_frame_logs, f, indent=2)
        logging.info('帧级日志已写入 %s', args.log_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

    scan_dir = args.scan_dir or osp.join(args.data_dir, KEYFRAME_SUBDIR)
    map_path = args.map_path or osp.join(args.data_dir, PRIOR_MAP_FILENAME)
    tum_path = args.tum_path or osp.join(args.data_dir, TUM_FILENAME)

    args.scan_dir = scan_dir
    args.map_path = map_path
    args.tum_path = tum_path

    if not osp.isdir(scan_dir):
        raise FileNotFoundError(f'Scan directory not found: {scan_dir}')
    if not osp.isfile(map_path):
        raise FileNotFoundError(f'Prior map not found: {map_path}')
    if not osp.isfile(tum_path):
        raise FileNotFoundError(f'TUM pose file not found: {tum_path}')
    if not osp.isfile(args.config):
        raise FileNotFoundError(f'Config file not found: {args.config}')

    logging.info('加载先验地图：%s', map_path)
    prior_map = load_point_cloud(map_path)
    logging.info('原始先验地图点数：%d', prior_map.shape[0])
    prior_map = voxel_downsample(prior_map, args.map_voxel_size)
    logging.info('体素下采样 (%.3fm) 后先验地图点数：%d', args.map_voxel_size, prior_map.shape[0])
    if prior_map.shape[0] == 0:
        raise RuntimeError('先验地图在体素下采样后为空，请检查数据。')

    poses = parse_tum_file(tum_path)
    if not poses:
        raise RuntimeError('未能从 TUM 文件解析到有效轨迹。')

    global_init_transform = None
    if INITIAL_TRANSFORM is not None:
        global_init_transform = np.asarray(INITIAL_TRANSFORM, dtype=np.float64).reshape(4, 4)
        logging.info('使用全局初始变换矩阵 (LIO->地图)，来自 INITIAL_TRANSFORM：\n%s', global_init_transform)
    elif INITIAL_POSE is not None:
        global_init_transform = pose_to_transform(INITIAL_POSE, radians=POSE_IN_RADIANS)
        logging.info('使用全局初始位姿矩阵 (LIO->地图)，来自 INITIAL_POSE=%s', str(INITIAL_POSE))

    # 默认同时跑 FPFH 与非 FPFH 两条流程，输出在不同子目录
    for use_fpfh in (True, False):
        run_mode(args, use_fpfh, prior_map, poses, global_init_transform)


if __name__ == '__main__':
    main()
