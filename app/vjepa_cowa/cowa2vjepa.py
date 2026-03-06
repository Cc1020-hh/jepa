# convert_to_vjepa_format.py

import os
import pickle as pkl
import numpy as np
import json
import h5py
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import cv2

def create_video_from_images(image_paths, output_path, fps=5):
    """从图片序列创建视频"""
    if not image_paths or len(image_paths) == 0:
        print(f"警告: 没有图片路径")
        return False
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"无法读取图片: {image_paths[0]}")
        return False
    
    height, width = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for img_path in tqdm(image_paths, desc=f"Creating video", leave=False):
        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
    
    out.release()
    return True

def convert_clip_to_vjepa_format(clip_id, clip_data, output_dir):
    """
    将单个clip转换为V-JEPA格式
    
    Args:
        clip_id: clip的ID
        clip_data: 从pkl加载的clip数据
        output_dir: 输出根目录
    """
    # 创建clip目录
    clip_dir = os.path.join(output_dir, f"trajectory_{clip_id}")
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(os.path.join(clip_dir, "recordings/MP4"), exist_ok=True)
    
    # 1. 获取相机列表
    camera_params = clip_data.get('camera_params', {})
    available_cameras = [cam for cam in camera_params.keys()]
    
    if not available_cameras:
        print(f"警告: Clip {clip_id} 没有相机参数")
        return None
    
    # 2. 使用第一个可用相机作为参考（通常是CAM_FRONT）
    reference_cam = None
    for preferred in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT']:
        if preferred in clip_data and len(clip_data[preferred]) > 0:
            reference_cam = preferred
            break
    
    if reference_cam is None:
        reference_cam = available_cameras[0]
        if reference_cam not in clip_data or len(clip_data[reference_cam]) == 0:
            print(f"警告: Clip {clip_id} 参考相机 {reference_cam} 没有数据")
            return None
    
    reference_frames = clip_data[reference_cam]
    T = len(reference_frames)
    
    print(f"Clip {clip_id}: 参考相机={reference_cam}, 帧数={T}")
    
    # 3. 提取车辆状态序列
    ego_poses = np.zeros((T, 7))  # [x, y, z, roll, pitch, yaw, velocity]
    timestamps = np.zeros(T)
    
    for t, frame_info in enumerate(reference_frames):
        # 位置
        ego_poses[t, :3] = frame_info['T_ego2global']
        
        # 旋转（从旋转矩阵转换为欧拉角）
        R_ego2global = frame_info['R_ego2global']
        euler = Rotation.from_matrix(R_ego2global).as_euler('xyz', degrees=False)
        ego_poses[t, 3:6] = euler
        
        # 速度（已经在数据中）
        ego_poses[t, 6] = frame_info.get('velocity', 0.0)
        
        # 时间戳
        timestamps[t] = frame_info['main_timestamp']
    # 数据预处理时改为存储相对坐标（以第0帧为原点）
    # origin_x = ego_poses[0, 0]
    # origin_y = ego_poses[0, 1]
    # origin_yaw = ego_poses[0, 5]

    # cos_h = np.cos(-origin_yaw)
    # sin_h = np.sin(-origin_yaw)

    # for t in range(T):
    #     dx = ego_poses[t, 0] - origin_x
    #     dy = ego_poses[t, 1] - origin_y
    #     ego_poses[t, 0] = cos_h * dx - sin_h * dy  # ego-relative x
    #     ego_poses[t, 1] = sin_h * dx + cos_h * dy  # ego-relative y
    #     ego_poses[t, 5] = ego_poses[t, 5] - origin_yaw  # relative yaw
    # 4. 创建HDF5文件
    h5_path = os.path.join(clip_dir, "trajectory.h5")
    with h5py.File(h5_path, 'w') as f:
        obs_group = f.create_group('observation')
        
        # 保存车辆状态
        robot_group = obs_group.create_group('robot_state')
        robot_group.create_dataset('cartesian_position', data=ego_poses[:, :6])
        robot_group.create_dataset('gripper_position', data=ego_poses[:, 6])
        robot_group.create_dataset('timestamps', data=timestamps)
        
        # 保存相机外参
        cam_group = obs_group.create_group('camera_extrinsics')
        for cam_name, cam_params in camera_params.items():
            # 获取外参
            T_cam2ego = np.array(cam_params['T_cam2ego'])
            R_cam2ego = np.array(cam_params['R_cam2ego'])
            
            # 转换为欧拉角形式 [x, y, z, roll, pitch, yaw, 0]
            euler = Rotation.from_matrix(R_cam2ego).as_euler('xyz', degrees=False)
            extrinsic_7d = np.concatenate([T_cam2ego, euler, [0]])
            
            # 对所有时间步重复相同的外参
            extrinsics = np.tile(extrinsic_7d, (T, 1))
            cam_group.create_dataset(f"{cam_name}_left", data=extrinsics)
    
    # 5. 创建视频文件
    video_metadata = {}
    for cam_name in available_cameras:
        if cam_name not in clip_data:
            continue
        
        cam_frames = clip_data[cam_name]
        if len(cam_frames) == 0:
            continue
        
        # 获取图片路径列表
        image_paths = [frame['img_path'] for frame in cam_frames]
        
        # 验证图片存在
        valid_paths = [p for p in image_paths if os.path.exists(p)]
        if len(valid_paths) < len(image_paths) * 0.9:  # 如果超过10%缺失
            print(f"警告: 相机 {cam_name} 缺失过多图片 ({len(valid_paths)}/{len(image_paths)})")
            continue
        
        # 创建视频
        video_path = os.path.join(clip_dir, f"recordings/MP4/{cam_name}.mp4")
        success = create_video_from_images(valid_paths, video_path, fps=5)
        
        if success:
            video_metadata[f"{cam_name.lower()}_mp4_path"] = f"recordings/MP4/{cam_name}.mp4"
    
    # 6. 创建metadata.json
    metadata = {
        "clip_id": clip_id,
        "duration": (timestamps[-1] - timestamps[0]) / 1e9,  # 纳秒转秒
        "frame_count": T,
        "average_velocity": float(clip_data.get('average_velocity', 0)),
        "max_velocity": float(clip_data.get('max_velocity', 0)),
        "min_velocity": float(clip_data.get('min_velocity', 0)),
        **video_metadata,
        "camera_params": camera_params
    }
    
    with open(os.path.join(clip_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return clip_dir

def process_all_clips(pkl_path, output_dir, max_clips=1200):
    """
    处理前 max_clips 个 clips 并转换为 V-JEPA 格式
    """
    # 加载数据
    print(f"加载数据: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        camera_data_all = pkl.load(f)

    # 如果是字典，获取所有 items 并限制数量
    all_items = list(camera_data_all.items())[:max_clips]
    print(f"原共有 {len(camera_data_all)} 个clips，本次将处理前 {len(all_items)} 个")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 创建样本列表
    sample_list = []

    # 处理限制数量后的 clips
    for clip_id, clip_data in tqdm(all_items, desc=f"Converting first {max_clips} clips"):
        try:
            clip_dir = convert_clip_to_vjepa_format(clip_id, clip_data, output_dir)
            if clip_dir:
                sample_list.append(clip_dir)
        except Exception as e:
            print(f"处理clip {clip_id} 时出错: {e}")
            continue

    # 保存样本列表
    list_path = os.path.join(output_dir, "train.txt")
    with open(list_path, 'w') as f:
        f.write('\n'.join(sample_list))

    print(f"\n完成! 共处理 {len(sample_list)} 个clips")
    return sample_list

if __name__ == '__main__':
    # 配置路径
    pkl_path = "/disk/deepdata/dataset/nvs/pkl/vjepa/clip_image_nvs_panorama_EQ5S/0-10000_4800.pkl"
    output_dir = "/disk/deepdata/dataset/vjepa_format_ad_data/clip_image_nvs_panorama_EQ5S/0-10000_4800"
    
    # 执行转换
    sample_list = process_all_clips(pkl_path, output_dir)
    
    print(f"\n数据转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"可用于训练的clips数量: {len(sample_list)}")
