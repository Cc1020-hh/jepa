# import bson
# from bson import ObjectId
# import lmdb
# import pickle 
import libclip_container
from pymongo import MongoClient
from tqdm import tqdm
from PIL import Image
import os
import numpy as np
# import math
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor, as_completed
# import functools
from collections import defaultdict
import cv2
import json
from pathlib import Path
# from scipy.spatial.transform import Rotation

camera_names = {
    "/camera/panorama/1/h264": "CAM_BACK_LEFT",
    "/camera/panorama/2/h264": "CAM_FRONT_LEFT",
    "/camera/panorama/3/h264": "CAM_FRONT",
    "/camera/panorama/4/h264": "CAM_FRONT_RIGHT",
    "/camera/panorama/5/h264": "CAM_BACK_RIGHT",
    "/camera/stereo/back/1/h264": "CAM_STEREO_BACK_1",
    "/camera/stereo/back/2/h264": "CAM_STEREO_BACK_2",
    "/camera/stereo/left/1/h264": "CAM_STEREO_LEFT_1",
    "/camera/stereo/left/2/h264": "CAM_STEREO_LEFT_2",
    "/camera/stereo/right/1/h264": "CAM_STEREO_RIGHT_1",
    "/camera/stereo/right/2/h264": "CAM_STEREO_RIGHT_2",
    "/camera/stereo/front/1/h264": "CAM_STEREO_FRONT_1",
    "/camera/stereo/front/2/h264": "CAM_STEREO_FRONT_2",
    "/camera/surround/front/h264": "CAM_SUR_FRONT",
    "/camera/surround/left/h264": "CAM_SUR_LEFT",
    "/camera/surround/right/h264": "CAM_SUR_RIGHT",
    "/camera/surround/back/h264": "CAM_SUR_BACK",
}

def compute_velocity_from_poses(positions, timestamps):
    """
    从位置序列和时间戳计算速度
    
    Args:
        positions: [T, 3] numpy数组，xyz位置
        timestamps: [T] 时间戳列表（单位：微秒）
    
    Returns:
        velocities: [T] 线速度（米/秒）
    """
    T = len(positions)
    velocities = np.zeros(T)
    
    # 将时间戳转换为秒
    timestamps = np.array(timestamps, dtype=np.float64)
    # 根据数值大小判断时间戳单位
    if timestamps[0] > 1e15:  # 纳秒 (19位数字)
        timestamps = timestamps / 1e9
        # print(f"检测到纳秒时间戳，转换为秒")
    elif timestamps[0] > 1e12:  # 微秒 (16位数字)
        timestamps = timestamps / 1e6
        print(f"检测到微秒时间戳，转换为秒")
    elif timestamps[0] > 1e9:  # 毫秒 (13位数字)
        timestamps = timestamps / 1e3
        print(f"检测到毫秒时间戳，转换为秒")
    # print(f"时间戳范围: {timestamps[0]:.3f}s ~ {timestamps[-1]:.3f}s")
    # print(f"总时长: {timestamps[-1] - timestamps[0]:.3f}s")
    for t in range(T - 1):
        dt = timestamps[t + 1] - timestamps[t]
        
        if dt <= 0 or dt > 1.0:  # 时间间隔异常（>1秒可能有问题）
            print(f"警告: 时间间隔异常 dt={dt:.3f}s at index {t}")
            velocities[t] = 0
            continue
        
        # 计算线速度 (v = Δs / Δt)
        position_diff = positions[t + 1] - positions[t]
        distance = np.linalg.norm(position_diff)
        velocities[t] = distance / dt
    
    # 最后一帧使用前一帧的值
    velocities[-1] = velocities[-2] if T > 1 else 0
    
    return velocities

def process_transform(transform):
    """处理相机变换参数"""
    x, y, z = transform[:3]  # 位置
    roll, pitch, yaw = transform[3:]  # 欧拉角（弧度）
    T_cam2ego = np.array([x, y, z])

    def euler_to_rotation_matrix(roll, pitch, yaw):
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(roll), -np.sin(roll)],
                        [0, np.sin(roll), np.cos(roll)]])

        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x 
        return R

    R_cam2ego = euler_to_rotation_matrix(roll, pitch, yaw)
    return T_cam2ego, R_cam2ego

def _get_match_dict(matched_ch, selected_topics, main_length, pop_miss=False):
    """获取匹配字典"""
    matched_dict = defaultdict(list)
    print("matched_dict[main_index]", len(selected_topics))
    
    for camera_topic in selected_topics:
        for matched_proto in matched_ch:
            if matched_proto.camera != camera_topic:
                continue
            print(matched_proto.camera)
            matched_index = 0
            matched_length = len(matched_proto.pairs)
            for main_index in range(main_length):
                if matched_index >= matched_length:
                    matched_dict[main_index].append(None)
                elif main_index == matched_proto.pairs[matched_index].lidar:
                    matched_dict[main_index].append(
                        matched_proto.pairs[matched_index].camera)
                    matched_index += 1
                elif main_index < matched_proto.pairs[matched_index].lidar:
                    matched_dict[main_index].append(None)

    if pop_miss:
        pop_key_list = []
        for key, value in matched_dict.items():
            if None in value:
                pop_key_list.append(key)
        for key in pop_key_list:
            matched_dict.pop(key)

    return matched_dict

def get_sampling_indices(total_frames, target_fps=5):
    """根据目标帧率计算需要采样的帧索引"""
    if total_frames == 0:
        return []

    original_fps = 10
    sampling_interval = original_fps // target_fps
    
    indices = list(range(0, total_frames, sampling_interval))
    return indices

def validate_clip_cameras(all_topics, required_cameras):
    """验证clip是否包含所有必需的相机"""
    found_cameras = []
    for required_cam in required_cameras:
        for topic in all_topics:
            if required_cam in topic:
                found_cameras.append(topic)
                break
    if len(found_cameras) < 2:
        print(found_cameras)
        return False, found_cameras
    else:
        return True, found_cameras

def pre_create_directories(save_dir, sensors_config):
    """预先创建所有可能需要的目录"""
    metadata_dir = os.path.join(save_dir, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    print(f"创建metadata目录: {metadata_dir}")
    for cam in sensors_config['cams']:
        for prefix in ['/camera/']:
            cam_path = prefix + cam + '/h264'
            cam_topic_clean = cam_path.lstrip('/')
            image_dir = os.path.join(save_dir, cam_topic_clean)
            os.makedirs(image_dir, exist_ok=True)
            print(f"{image_dir} have created")
    print("目录预创建完成")

def save_image_optimized(image, save_path):
    """优化的图像保存函数"""
    if os.path.exists(save_path):
        return False
    
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif image.shape[2] == 4:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB))
        else:
            img = Image.fromarray(image)
    else:
        img = Image.fromarray(image)
        
    if img.size != (1920, 1080) and img.size != (1280,800):
        print(f"警告: 图像尺寸不匹配 - 期望: (1920, 1080), 实际: {img.size}")
        return False
    
    img.save(save_path, format='PNG', optimize=False, compress_level=1)
    return True

def process_single_clip_worker_with_match_optimized(clip_info, save_dir, sensors_config):
    """优化版本的clip处理函数（新增速度计算）"""
    try:
        clip_path = clip_info['data_path']
        clip_id = clip_info['_id']
        
        # 检查是否已经处理过
        metadata_dir = os.path.join(save_dir, 'metadata')
        metadata_path = os.path.join(metadata_dir, f"metadata_{clip_id}.json")
        if os.path.exists(metadata_path):
            print(f"Clip {clip_id} 已处理，跳过")
            return {
                'clip_id': str(clip_id),
                'metadata_path': metadata_path,
                'frame_count': 0,
                'success': True,
                'skipped': True
            }
        
        clip = libclip_container.DataClipReader(clip_path)
        clip.ReadDriveData()
        all_topics = clip.ListTopics()
        
        # 检查pose provider
        pose_flag = False
        for topic in all_topics:
            if '/pose/odom' in topic:
                pose_provider = libclip_container.PoseProvider(
                    clip.GetChannel('/pose/odom'))
                pose_flag = True
                break
        
        if not pose_flag:
            print(f'Clip {clip_id}: pose_provider not found') 
            return None
        
        # 检查match通道
        if '/match' not in all_topics:
            print(f'Clip {clip_id}: /match channel not found')
            return None
        
        # 验证是否包含所有必需的相机
        has_all_cameras, selected_camera_topics = validate_clip_cameras(
            all_topics, sensors_config['cams'])
        if not has_all_cameras:
            print(f'Clip {clip_id}: 不包含所有必需的相机，跳过')
            return None
            
        # 获取主传感器（lidar）长度
        main_ch = clip.GetChannel(sensors_config['main'])
        main_length = len(main_ch)
        
        # 获取匹配字典
        matched_ch = clip.GetChannel('/match')
        image_match_dict = _get_match_dict(
            matched_ch, selected_camera_topics, main_length, pop_miss=True
        )
        
        if not image_match_dict:
            print(f'Clip {clip_id}: no matched frames found')
            return None

        # 对匹配的帧进行采样
        matched_main_indices = list(image_match_dict.keys())
        sampling_indices = get_sampling_indices(len(matched_main_indices), target_fps=5)
        sampled_main_indices = [matched_main_indices[i] for i in sampling_indices 
                               if i < len(matched_main_indices)]
        
        print(f"Clip {clip_id}: 包含所有相机，主传感器总帧数 {main_length}, "
              f"匹配帧数 {len(matched_main_indices)}, "
              f"采样后帧数 {len(sampled_main_indices)}")
        
        # ========== 新增：收集位置和时间戳用于计算速度 ==========
        ego_positions_list = []
        ego_timestamps_list = []
        # ====================================================
        
        frame_info_list = []
        camera_params = {}
        
        # 获取相机通道和参数
        camera_channels = {}
        intrinsics_data = clip.GetIntrinsics()
        
        for cam_topic in selected_camera_topics:
            camera_channels[cam_topic] = clip.GetChannel(cam_topic)
            cam = camera_names.get(cam_topic)
            
            if cam is None:
                continue
                
            if cam not in camera_params:
                if cam_topic in intrinsics_data:
                    intrinsics = intrinsics_data[cam_topic][0].reshape(3, 3)
                    distortion = intrinsics_data[cam_topic][1].flatten()
                    
                    camera_ch = camera_channels[cam_topic]
                    if len(camera_ch) > 0:
                        transform = camera_ch[0].transform
                        if transform is not None:
                            transform = camera_ch[0].transform
                            if transform is not None:
                                T_cam2ego, R_cam2ego = process_transform(transform)
                                camera_params[cam] = {
                                    "intrinsics": intrinsics.tolist(),
                                    "distortion": distortion.tolist(),
                                    "T_cam2ego": T_cam2ego.tolist(),
                                    "R_cam2ego": R_cam2ego.tolist(),
                                }
        processed_frames = 0
        for main_index in sampled_main_indices:
            if main_index not in image_match_dict:
                continue
                
            camera_indices = image_match_dict[main_index]
            
            if len(camera_indices) != len(selected_camera_topics) or None in camera_indices:
                continue
                
            main_timestamp = main_ch[main_index].timestamp
            pose = pose_provider.GetTransforms(main_timestamp)
            T_ego2global = pose[:3, 3]
            R_ego2global = pose[:3, :3]
            
            # ========== 新增：收集位置和时间戳 ==========
            ego_positions_list.append(T_ego2global.copy())
            ego_timestamps_list.append(main_timestamp)
            # ==========================================
            
            frame_valid = True  
            frame_cameras = {}
            
            # 处理每个相机
            for i, cam_topic in enumerate(selected_camera_topics):
                camera_index = camera_indices[i]
                camera_ch = camera_channels[cam_topic]
                
                if camera_index >= len(camera_ch):
                    frame_valid = False
                    break
                
                cam = camera_names.get(cam_topic)
                if cam is None:
                    frame_valid = False
                    break
                
                try:
                    image = camera_ch.GetMat(camera_index)
                    
                    cam_topic_clean = cam_topic.lstrip('/')
                    image_dir = os.path.join(save_dir, cam_topic_clean)
                    
                    camera_timestamp = camera_ch[camera_index].timestamp
                    save_path = os.path.join(image_dir, 
                                           f"{main_timestamp}_{main_index}_{camera_index}.png")
                    
                    save_image_optimized(image, save_path)
                    
                    frame_cameras[cam] = {
                        "img_path": save_path,
                        "T_ego2global": T_ego2global.tolist(),
                        "R_ego2global": R_ego2global.tolist(),
                        "main_timestamp": int(main_timestamp),
                        "camera_timestamp": int(camera_timestamp),
                        "main_index": int(main_index),
                        "camera_index": int(camera_index)
                    }
                    
                except Exception as e:
                    print(f"处理相机 {cam} 帧 {camera_index} 时出错: {e}")
                    frame_valid = False
                    break
            
            if frame_valid:
                frame_info_list.append(frame_cameras)
                processed_frames += 1
        
        if processed_frames == 0:
            print(f'Clip {clip_id}: 没有成功处理任何帧')
            return None
        
        # ========== 新增：计算速度 ==========
        ego_positions = np.array(ego_positions_list)
        ego_timestamps = np.array(ego_timestamps_list)
        velocities = compute_velocity_from_poses(ego_positions, ego_timestamps)
        
        # 将速度信息添加到每一帧
        for idx, frame_cameras in enumerate(frame_info_list):
            for cam_name in frame_cameras.keys():
                frame_cameras[cam_name]['velocity'] = float(velocities[idx])
        # ===================================
        
        # 保存元数据到JSON文件
        result_data = {
            'clip_id': str(clip_id),
            'camera_params': camera_params,
            'frames': frame_info_list,
            'frame_count': processed_frames,
            # ========== 新增：速度统计信息 ==========
            'average_velocity': float(np.mean(velocities)),
            'max_velocity': float(np.max(velocities)),
            'min_velocity': float(np.min(velocities)),
            # ======================================
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(result_data, f)
        
        print(f"✓ Clip {clip_id}: {processed_frames}帧, "
              f"平均速度 {result_data['average_velocity']:.2f} m/s")
        
        return {
            'clip_id': str(clip_id),
            'metadata_path': metadata_path,
            'frame_count': processed_frames,
            'success': True,
            'skipped': False
        }
        
    except Exception as e:
        print(f"处理clip {clip_info.get('_id', 'unknown')} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def query_vaild_clips(wanted_start, wanted_end, vechile_type, data_version):
    """查询有效的clips"""
    client = MongoClient('mongodb://e2e_data_platform_r:DAT33pl0tFReadpr0d@mongo.cowarobot.cn:27017/e2e-data-platform-prod?tls=false')
    db = client['e2e-data-platform-prod']
    collection = db['clip']
    valid_clips_data = collection.find(
        {"truth.version": f"{data_version}", "vehicle.type": f"{vechile_type}"}
    )
    res = []
    print("******💐 💐 💐 💐 ********，斯黛拉正在获取资源")
    for clips_idx, clips in enumerate(valid_clips_data[wanted_start:wanted_end]): 
        prefix_path = "/disk/deepdata/clipground-prod/"
        clip_id = clips["_id"]
        storage = clips["storage"]
        if storage == 'e2e':
            prefix_path = "/disk/e2e/clipground-prod/"
        try:
            clips_msg = collection.find_one({"_id": clip_id}, {
                "source_id",
                "data_path"
            })
            clip_info = {
                "_id": clip_id,
                "source_id": clips_msg['source_id'],
                "data_path": prefix_path + clips_msg['data_path']
            }
            res.append(clip_info)
        except:
            print(f"No query result of _id: {clip_id}")
    print(f"******💐 💐 💐 💐 ********，斯黛拉获取资源完毕,总资源为{len(res)}")
    client.close()
    return res

def process_single_metadata_file(metadata_path):
    """处理单个元数据文件"""
    try:
        with open(metadata_path, 'r') as f:
            clip_data = json.load(f)
            clip_id = clip_data['clip_id']
            
            clip_result = {
                'clip_id': clip_id,
                'camera_params': clip_data['camera_params'],
                'valid_frames': [],
                'missing_images': [],
                # ========== 新增：速度统计 ==========
                'average_velocity': clip_data.get('average_velocity', 0.0),
                'max_velocity': clip_data.get('max_velocity', 0.0),
                'min_velocity': clip_data.get('min_velocity', 0.0),
                # ==================================
            }
            
            for cam in camera_names.values():
                clip_result[cam] = []
            
            for frame in clip_data['frames']:
                frame_valid = True
                frame_data = {}
                
                for cam, frame_info in frame.items():
                    img_path = frame_info['img_path']
                    
                    if not os.path.exists(img_path):
                        clip_result['missing_images'].append(img_path)
                        frame_valid = False
                        break
                    
                    if os.path.getsize(img_path) < 1000:
                        print(f"警告: 图片文件过小 {img_path}")
                        clip_result['missing_images'].append(img_path)
                        frame_valid = False
                        break
                    
                    frame_info['T_ego2global'] = np.array(frame_info['T_ego2global'])
                    frame_info['R_ego2global'] = np.array(frame_info['R_ego2global'])
                    # ========== 速度已经在JSON中 ==========
                    # frame_info['velocity'] 保持不变
                    # ======================================
                    frame_data[cam] = frame_info
                
                if frame_valid:
                    for cam, frame_info in frame_data.items():
                        clip_result[cam].append(frame_info)
                    clip_result['valid_frames'].append(True)
            
            return clip_result
            
    except Exception as e:
        print(f"读取元数据文件失败 {metadata_path}: {e}")
        return None

def merge_camera_data_from_json(results, save_dir, second_run=None, num_processes=None):
    """从JSON元数据文件合并结果（多进程版本）"""
    if num_processes is None:
        num_processes = min(os.cpu_count() - 1, 32)
    
    print(f"使用 {num_processes} 个进程合并数据...")
    
    metadata_paths = []
    skipped_clips = 0
    
    for result in results:
        if result is None or not result['success']:
            continue
        if not second_run and result.get('skipped', False):
            skipped_clips += 1
            continue
        metadata_paths.append(result['metadata_path'])
    
    print(f"需要合并 {len(metadata_paths)} 个元数据文件，跳过 {skipped_clips} 个")
    
    merged_camera_data = {}
    all_missing_images = []
    total_frames = 0
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(process_single_metadata_file, path): path 
            for path in metadata_paths
        }
        
        with tqdm(total=len(metadata_paths), desc='Merging metadata') as pbar:
            for future in as_completed(futures):
                try:
                    clip_result = future.result()
                    if clip_result:
                        clip_id = clip_result['clip_id']
                        
                        merged_camera_data[clip_id] = {
                            'camera_params': clip_result['camera_params'],
                            # ========== 新增：保存速度统计 ==========
                            'average_velocity': clip_result['average_velocity'],
                            'max_velocity': clip_result['max_velocity'],
                            'min_velocity': clip_result['min_velocity'],
                            # ======================================
                        }
                        
                        frame_count = 0
                        for cam in camera_names.values():
                            merged_camera_data[clip_id][cam] = clip_result[cam]
                            frame_count = max(frame_count, len(clip_result[cam]))
                        
                        total_frames += frame_count
                        
                        if clip_result['missing_images']:
                            all_missing_images.extend(clip_result['missing_images'])
                            
                except Exception as e:
                    print(f"合并任务时出错: {e}")
                pbar.update(1)
    
    if all_missing_images:
        missing_log = os.path.join(save_dir, 'missing_images_during_merge.txt')
        with open(missing_log, 'w') as f:
            f.write('\n'.join(all_missing_images))
        print(f"警告: 发现 {len(all_missing_images)} 个缺失的图片，列表已保存到 {missing_log}")
    
    total_clips = len(merged_camera_data)
    print(f"合并完成: 处理 {total_clips} 个clips, 跳过 {skipped_clips} 个已处理clips, "
          f"总共 {total_frames} 帧")
    
    return merged_camera_data, total_clips, total_frames


if __name__ == '__main__':
    sensors_config = dict(
        main='/main/ruby/lidar_points',
        pose='/pose/odom',
        cams=[
            # "panorama/1",
            # "panorama/2",
            # "panorama/3", 
            # "panorama/4", 
            # "panorama/5",
            # "surround/front",
            # "surround/left",
            # "surround/right",
            # "surround/back",
        ]
    )
    vechile_type = 'EQ5S'
    data_version = '1.0.9'
    wanted_start = 0
    wanted_end = 10000
    camera_type = sensors_config['cams'][0].split('/')[0]
    task = 'vjepa'
    data_root = Path('/disk/deepdata/dataset/nvs')
    save_dir = data_root / f'data/{task}/clip_image_nvs_{camera_type}_{vechile_type}/{wanted_start}-{wanted_end}'
    
    pre_create_directories(save_dir, sensors_config)
    
    res = query_vaild_clips(wanted_start, wanted_end, vechile_type, data_version)
    # for clip_info in res:
    #     process_single_clip_worker_with_match_optimized(clip_info, save_dir, sensors_config)
    num_processes = min(os.cpu_count() - 1, 96) 
    print(f"使用 {num_processes} 个进程进行处理")
    # res = res[:4800]
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(
                process_single_clip_worker_with_match_optimized,
                clip_info, save_dir, sensors_config
            ): clip_info for clip_info in res
        }
        with tqdm(total=len(res), desc='Processing clips') as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理任务时出错: {e}")
                pbar.update(1)
    
    print("正在合并结果...")
    second_run = False
    results = results[9000:10000]
    camera_data, clip_number, frame_number = merge_camera_data_from_json(
        results, save_dir, second_run=True, num_processes=96
    )
    output_dir = Path(data_root / f"pkl/{task}/clip_image_nvs_{camera_type}_{vechile_type}")
    os.makedirs(output_dir,exist_ok=True)
    output_path = output_dir / f"{wanted_start}-{wanted_end}_val_{len(results)}.pkl"
    
    with open(output_path, "wb") as f:
        pkl.dump(camera_data, f)
    
    print(f"******💐 💐 💐 💐 ********，斯黛拉已经长大成为黛丽拉，"
          f"摧毁部落{clip_number}个，杀死敌人数量{frame_number}")
    print(f"结果已保存到: {output_path}")
    
    # ========== 新增：打印速度统计信息 ==========
    # print("\n========== 速度统计 ==========")
    # all_avg_velocities = []
    # for clip_id, clip_data in camera_data.items():
    #     avg_vel = clip_data.get('average_velocity', 0)
    #     all_avg_velocities.append(avg_vel)
    #     print(f"Clip {clip_id}: 平均速度 {avg_vel:.2f} m/s, "
    #           f"最大 {clip_data.get('max_velocity', 0):.2f} m/s")
    
    # if all_avg_velocities:
    #     print(f"\n整体平均速度: {np.mean(all_avg_velocities):.2f} m/s")
    #     print(f"速度标准差: {np.std(all_avg_velocities):.2f} m/s")
    # print("==============================\n")
    # ==========================================
