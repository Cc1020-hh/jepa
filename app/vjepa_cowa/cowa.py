# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import json
import os
from logging import getLogger
from math import ceil
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2
import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
_GLOBAL_SEED = 0
logger = getLogger()
import time
import json
import os
import h5py
import numpy as np
import torch
import torch.utils.data
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation
from app.vjepa_cowa.vis_mask import visualize_masks
class AutonomousDrivingHFTemporalDataset(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 支持历史帧+未来帧，兼容 Processor"""
    
    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        num_context_frames=2,
        num_target_frames=16,
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
        processor=None,  # 新增
    ):
        self.data_path = data_path
        self.camera_views = camera_views
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.total_frames = num_context_frames + num_target_frames - 1
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        self.processor = processor  # 优先使用 processor
        
        with open(data_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        print(f"加载了 {len(self.samples)} 个训练样本")
        print(f"Context帧数: {num_context_frames}, Target帧数: {num_target_frames}")
        print(f"使用 {'Processor' if processor else 'Transform'}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        
        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"加载失败 {path} (retry {retry+1}/{max_retries}): {e}")
                    index = np.random.randint(len(self))
                    path = self.samples[index]
                else:
                    raise e
    
    def load_clip(self, path):
        """加载单个clip的数据"""
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
        
        # 4. 加载视频
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        vlen = len(vr)
        
        # 5. 计算采样参数
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.total_frames * fstp
        
        if vlen < nframes:
            nframes = vlen
            fstp = max(1, vlen // self.total_frames)
        
        # 6. 随机采样
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.total_frames]
        
        if len(indices) < self.total_frames:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.total_frames - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        # 7. 读取视频帧
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T_total, H, W, C]
        
        # 8. 分离context和target
        context_buffer = buffer[:self.num_context_frames]  # [2, H, W, C]
        target_buffer = buffer[self.num_context_frames-1:
                              self.num_context_frames-1+self.num_target_frames]  # [16, H, W, C]
        
        # 9. 数据处理 - 关键修复点
        if self.processor is not None:
            # 使用 HuggingFace Processor
            # 注意：processor 期望 PIL Image 或 numpy array [H, W, C]
            # Context frames: [2, H, W, C]
            context_processed = self.processor(
                videos=context_buffer,  # 改为 videos 参数
                return_tensors="pt"
            )
            
            # Target frames: [16, H, W, C]
            target_processed = self.processor(
                videos=target_buffer,  # 改为 videos 参数
                return_tensors="pt"
            )
            
            # 提取像素值并移除多余的batch维度
            # processor 可能返回 [1, T, C, H, W] 或 [T, C, H, W]
            context_buffer = context_processed["pixel_values_videos"]
            target_buffer = target_processed["pixel_values_videos"]
            
            # 确保格式是 [T, C, H, W]
            if context_buffer.dim() == 5:  # [1, T, C, H, W]
                context_buffer = context_buffer.squeeze(0)
            if target_buffer.dim() == 5:
                target_buffer = target_buffer.squeeze(0)
                
        elif self.transform is not None:
            # 使用自定义 transform
            # breakpoint()
            full_buffer = np.concatenate([context_buffer, target_buffer], axis=0)
            full_buffer = self.transform(full_buffer)  # 应返回 [T, C, H, W] tensor
            
            # 检查格式
            if isinstance(full_buffer, torch.Tensor):
                if full_buffer.shape[-1] == 3:  # [T, H, W, C]
                    full_buffer = full_buffer.permute(0, 3, 1, 2)
            else:
                if full_buffer.shape[-1] == 3:
                    full_buffer = np.transpose(full_buffer, (0, 3, 1, 2))
                full_buffer = torch.from_numpy(full_buffer).float()
            
            context_buffer = full_buffer[:self.num_context_frames]
            target_buffer = full_buffer[self.num_context_frames:
                                       self.num_context_frames+self.num_target_frames]
        else:
            # 手动转换
            context_buffer = torch.from_numpy(context_buffer).permute(0, 3, 1, 2).float() / 255.0
            target_buffer = torch.from_numpy(target_buffer).permute(0, 3, 1, 2).float() / 255.0
        
        # 10. 提取状态和外参
        context_indices = indices[:self.num_context_frames][::self.frameskip]
        context_states = states[context_indices]
        context_extrinsics = extrinsics[context_indices]
        
        target_indices = indices[self.num_context_frames-1:
                                self.num_context_frames-1+self.num_target_frames][::self.frameskip]
        target_states = states[target_indices]
        target_extrinsics = extrinsics[target_indices]
        # 11. 计算动作
        actions = self.compute_actions(target_states)
        actions_context = self.compute_actions(context_states)
        return {
            'context_frames': context_buffer,        # [2, C, H, W]
            'target_frames': target_buffer,          # [16, C, H, W]
            'context_states': context_states,
            'target_states': target_states,
            'context_extrinsics': context_extrinsics,
            'target_extrinsics': target_extrinsics,
            'actions': actions,
            'actions_context':actions_context,
            'context_indices': context_indices,
            'target_indices': target_indices,
        }
    
    def compute_actions(self, states):
        """计算动作序列"""
        T = len(states)
        actions = np.zeros((T - 1, 7))
        
        for t in range(T - 1):
            xyz_diff = states[t + 1, :3] - states[t, :3]
            
            R1 = Rotation.from_euler('xyz', states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler('xyz', states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler('xyz')
            
            velocity = states[t, 6]
            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])
        
        return actions


def autonomous_hf_driving_collate_fn(batch):
    """
    自定义collate函数 - 修复版
    """
    context_frames = torch.stack([item['context_frames'] for item in batch])
    target_frames = torch.stack([item['target_frames'] for item in batch])
    actions = torch.stack([torch.from_numpy(item['actions']) for item in batch])
    actions_context = torch.stack([torch.from_numpy(item['actions_context']) for item in batch])
    target_states = torch.stack([torch.from_numpy(item['target_states']) for item in batch])
    context_states = torch.stack([torch.from_numpy(item['context_states']) for item in batch])
    target_extrinsics = torch.stack([torch.from_numpy(item['target_extrinsics']) for item in batch])
    context_extrinsics = torch.stack([torch.from_numpy(item['context_extrinsics']) for item in batch])
    # 检查并确保格式正确
    # 应该已经是 [B, T, C, H, W] 格式
    assert context_frames.dim() == 5, f"Expected 5D tensor, got {context_frames.dim()}D"
    assert context_frames.shape[2] == 3, \
        f"Expected 3 channels at dim 2, got shape {context_frames.shape}"
    
    # 转换为 [B, C, T, H, W]
    context_frames = context_frames.permute(0, 2, 1, 3, 4)
    target_frames = target_frames.permute(0, 2, 1, 3, 4)
    
    return context_frames, target_frames, actions, target_states, target_extrinsics,actions_context,context_states,context_extrinsics


def init_data_hf_temporal(
    data_path,
    batch_size,
    num_context_frames=2,
    num_target_frames=16,
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    processor=None,  # 新增
    **kwargs
):
    """初始化数据加载器"""
    dataset = AutonomousDrivingHFTemporalDataset(
        data_path=data_path,
        processor=processor,  # 传入 processor
        num_context_frames=num_context_frames,
        num_target_frames=num_target_frames,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get('tubelet_size', 2),
        camera_frame=kwargs.get('camera_frame', False),
    )
    
    print(f"创建数据集: {len(dataset)} 个样本")
    print(f"Context帧数: {num_context_frames}, Target帧数: {num_target_frames}")
    
    if collator is None:
        collator = autonomous_hf_driving_collate_fn
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    
    print(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")
    
    return loader, sampler

def autonomous_driving_collate_fn(batch):
    """
    自定义collate函数，处理字典格式的batch
    
    Args:
        batch: List of dictionaries from dataset
    
    Returns:
        Tuple of (context_frames, target_frames, actions, target_states, target_extrinsics)
    """
    context_frames = torch.stack([item['context_frames'] for item in batch])
    target_frames = torch.stack([item['target_frames'] for item in batch])
    actions = torch.stack([torch.from_numpy(item['actions']) for item in batch])
    target_states = torch.stack([torch.from_numpy(item['target_states']) for item in batch])
    target_extrinsics = torch.stack([torch.from_numpy(item['target_extrinsics']) for item in batch])
    
    # 转换维度: [B, T, H, W, C] -> [B, C, T, H, W]
    context_frames = context_frames.permute(0, 4, 1, 2, 3)
    target_frames = target_frames.permute(0, 4, 1, 2, 3)
    
    return context_frames, target_frames, actions, target_states, target_extrinsics
def init_data_temporal(
    data_path,
    batch_size,
    num_context_frames=2,      # 新增参数
    num_target_frames=16,      # 新增参数
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    processor=None,  # 新增
    **kwargs
):
    """
    初始化自动驾驶数据加载器
    """
    dataset = AutonomousDrivingTemporalDataset(
        data_path=data_path,
        processor=processor,  # 传入
        num_context_frames=num_context_frames,
        num_target_frames=num_target_frames,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get('tubelet_size', 2),
        camera_frame=kwargs.get('camera_frame', False),
    )
    
    print(f"创建数据集: {len(dataset)} 个样本")
    print(f"Context帧数: {num_context_frames}, Target帧数: {num_target_frames}")
    
    # 使用自定义collate函数
    if collator is None:
        collator = autonomous_driving_collate_fn
    
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # 数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    
    print(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")
    
    return loader, sampler


class AutonomousDrivingTemporalDataset(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 支持历史帧+未来帧"""
    
    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        num_context_frames=2,      # 历史帧数（包括当前帧）
        num_target_frames=16,      # 未来帧数
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
    ):
        """
        Args:
            data_path: train.txt文件路径
            camera_views: 要使用的相机列表
            num_context_frames: context帧数（encoder输入）
            num_target_frames: target帧数（target_encoder输入）
            fps: 目标帧率
            transform: 数据增强
            camera_frame: 是否转换到相机坐标系
            frameskip: 帧间隔（tubelet_size）
        """
        self.data_path = data_path
        self.camera_views = camera_views
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.total_frames = num_context_frames + num_target_frames - 1  # 总共需要的帧数
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        
        # 加载样本列表
        with open(data_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        print(f"加载了 {len(self.samples)} 个训练样本")
        print(f"Context帧数: {num_context_frames}, Target帧数: {num_target_frames}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        
        # 尝试加载数据（失败则随机重试）
        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"加载失败 {path} (retry {retry+1}/{max_retries}): {e}")
                    index = np.random.randint(len(self))
                    path = self.samples[index]
                else:
                    raise e
    
    def load_clip(self, path):
        """加载单个clip的数据"""
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            
            # 加载相机外参
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
        
        # 4. 加载视频
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        vlen = len(vr)
        
        # 5. 计算采样参数
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.total_frames * fstp
        
        if vlen < nframes:
            # 如果视频太短，调整采样
            nframes = vlen
            fstp = max(1, vlen // self.total_frames)
        
        # 6. 随机采样时间窗口（确保有足够的帧）
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.total_frames]
        
        # 如果帧数不足，填充最后一帧
        if len(indices) < self.total_frames:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.total_frames - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        # 7. 读取视频帧
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T_total, H, W, C]
        
        # 8. 分离context帧和target帧
        # Context: 前2帧（历史帧+当前帧）
        # Target: 当前帧开始的16帧（包括当前帧）
        context_buffer = buffer[:self.num_context_frames]  # [2, H, W, C]
        target_buffer = buffer[self.num_context_frames-1:self.num_context_frames-1+self.num_target_frames]  # [16, H, W, C]
        
        # 9. 应用数据增强（对两部分分别增强，但使用相同的参数保持一致性）
        if self.transform is not None:
            # 合并后一起transform以保持一致性
            full_buffer = np.concatenate([context_buffer, target_buffer], axis=0)
            full_buffer = self.transform(full_buffer)
            # 再分离
            context_buffer = full_buffer[:self.num_context_frames]
            target_buffer = full_buffer[self.num_context_frames:self.num_context_frames+self.num_target_frames]
        
        # 10. 提取对应的状态和外参
        # Context状态：前2帧
        context_indices = indices[:self.num_context_frames][::self.frameskip]
        context_states = states[context_indices]
        context_extrinsics = extrinsics[context_indices]
        
        # Target状态：从当前帧开始的16帧
        target_indices = indices[self.num_context_frames-1:self.num_context_frames-1+self.num_target_frames][::self.frameskip]
        target_states = states[target_indices]
        target_extrinsics = extrinsics[target_indices]
        # 11. 计算动作（基于target状态）
        actions = self.compute_actions(target_states)
        
        return {
            'context_frames': context_buffer,        # [2, H, W, C]
            'target_frames': target_buffer,          # [16, H, W, C]
            'context_states': context_states,        # [2, 7]
            'target_states': target_states,          # [16, 7]
            'context_extrinsics': context_extrinsics,  # [2, 7]
            'target_extrinsics': target_extrinsics,    # [16, 7]
            'actions': actions,                      # [15, 7]
            'context_indices': context_indices,
            'target_indices': target_indices,
        }
    
    def compute_actions(self, states):
        """
        计算动作序列（状态差分）
        
        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        
        Returns:
            actions: [T-1, 7]
        """
        T = len(states)
        actions = np.zeros((T - 1, 7))
        
        for t in range(T - 1):
            # 位置差分
            xyz_diff = states[t + 1, :3] - states[t, :3]
            
            # 旋转差分
            R1 = Rotation.from_euler('xyz', states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler('xyz', states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler('xyz')
            
            # 使用当前帧的速度
            velocity = states[t, 6]
            
            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])
        
        return actions


def autonomous_driving_collate_fn_origin(batch):
    """
    自定义collate函数，处理字典格式的batch
    
    Args:
        batch: List of dictionaries from dataset
        buffer, actions, states, extrinsics, indices
    Returns:
        Tuple of (context_frames, target_frames, actions, target_states, target_extrinsics)
    """
    context_frames = torch.stack([item['buffer'] for item in batch])
    actions = torch.stack([torch.from_numpy(item['actions']) for item in batch])
    states = torch.stack([torch.from_numpy(item['states']) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item['extrinsics']) for item in batch])
    
    # 转换维度: [B, T, H, W, C] -> [B, C, T, H, W]
    # context_frames = context_frames.permute(0, 4, 1, 2, 3)
    # breakpoint()
    return context_frames, actions, states, extrinsics
def init_data(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    **kwargs
):
    """
    初始化自动驾驶数据加载器
    
    Args:
        data_path: train.txt文件路径
        batch_size: 批次大小
        frames_per_clip: 每个clip的帧数
        fps: 目标帧率
        camera_views: 相机列表
        transform: 数据增强
        collator: 数据整理函数
        num_workers: 工作进程数
        pin_mem: 是否使用pinned memory
        persistent_workers: 是否保持worker存活
        rank: 分布式训练rank
        world_size: 分布式训练world size
    """
    dataset = AutonomousDrivingDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get('tubelet_size', 2),
        camera_frame=kwargs.get('camera_frame', False),
    )
    
    print(f"创建数据集: {len(dataset)} 个样本")
    # 使用自定义collate函数
    if collator is None:
        collator = autonomous_driving_collate_fn_origin
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # 数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    
    print(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")
    
    return loader, sampler
class AutonomousDrivingDataset(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 兼容V-JEPA训练"""
    
    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
    ):
        """
        Args:
            data_path: train.txt文件路径
            camera_views: 要使用的相机列表
            frames_per_clip: 每个clip的帧数
            fps: 目标帧率
            transform: 数据增强
            camera_frame: 是否转换到相机坐标系
            frameskip: 帧间隔（tubelet_size）
        """
        self.data_path = data_path
        self.camera_views = camera_views
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        
        # 加载样本列表
        with open(data_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        print(f"加载了 {len(self.samples)} 个训练样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        
        # 尝试加载数据（失败则随机重试）
        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"加载失败 {path} (retry {retry+1}/{max_retries}): {e}")
                    index = np.random.randint(len(self))
                    path = self.samples[index]
                else:
                    raise e
    
    def load_clip(self, path):
        """加载单个clip的数据"""
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            
            # 加载相机外参
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
        
        # 4. 加载视频
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        vlen = len(vr)
        
        # 5. 计算采样参数
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.frames_per_clip * fstp
        
        if vlen < nframes:
            # 如果视频太短，使用所有帧
            nframes = vlen
            fstp = max(1, vlen // self.frames_per_clip)
        
        # 6. 随机采样时间窗口
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.frames_per_clip]
        
        # 如果帧数不足，填充最后一帧
        if len(indices) < self.frames_per_clip:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.frames_per_clip - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        # 7. 读取视频帧
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        
        # 8. 应用数据增强
        if self.transform is not None:
            buffer = self.transform(buffer)
        
        # 9. 提取对应的状态和外参
        states = states[indices][::self.frameskip]
        extrinsics = extrinsics[indices][::self.frameskip]
        
        # 10. 计算动作（状态差分）
        actions = self.compute_actions(states)
        return {
            'buffer': buffer,        # [2, H, W, C]
            'actions': actions,          # [16, H, W, C]
            'states': states,        # [2, 7]
            'extrinsics': extrinsics,          # [16, 7]
            'indices': indices,  # [2, 7]
        }
    
        return buffer, actions, states, extrinsics, indices
    
    def compute_actions(self, states):
        """
        计算动作序列（状态差分）
        
        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        
        Returns:
            actions: [T-1, 7]
        """
        T = len(states)
        actions = np.zeros((T - 1, 7))
        
        for t in range(T - 1):
            # 位置差分
            xyz_diff = states[t + 1, :3] - states[t, :3]
            
            # 旋转差分
            R1 = Rotation.from_euler('xyz', states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler('xyz', states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler('xyz')
            
            # 使用当前帧的速度
            velocity = states[t, 6]
            
            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])
        
        return actions

# ==================== 修改数据collate函数 ====================
def autonomous_driving_collate_fn_with_seg(batch):
    """处理分割标注的collate函数"""
    context_frames = torch.stack([item['buffer'] for item in batch])
    actions = torch.stack([torch.from_numpy(item['actions']) for item in batch])
    states = torch.stack([torch.from_numpy(item['states']) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item['extrinsics']) for item in batch])
    
    # 处理分割标注
    seg_targets = []
    for item in batch:
        seg_mask = item.get('seg_masks', None)  # [T, N, H, W]
        
        if seg_mask is not None and seg_mask.ndim == 4 and seg_mask.shape[1] > 0 and seg_mask.shape[0] > 0:
            # 取第一帧的所有实例
            masks_t0 = seg_mask[0]  # [N, H, W]
            N = masks_t0.shape[0]
            
            # 创建labels（全部为前景类0）
            labels = torch.ones(N, dtype=torch.long)
            
            target = {
                'labels': labels,
                'masks': masks_t0,  # [N, H, W]
            }
        else:
            # 空target
            target = {
                'labels': torch.zeros(0, dtype=torch.long),
                'masks': torch.zeros(0, context_frames.shape[3], context_frames.shape[4]),
            }
        
        seg_targets.append(target)
    
    return context_frames, actions, states, extrinsics, seg_targets
    
def init_data_seg(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    load_segmentation=True,  # 新增
    seg_data_root="/disk/deepdata/dataset/nvs/data/sam3_autolabeling_point",  # 新增
    crop_size = 256,
    **kwargs
):
    """
    初始化自动驾驶数据加载器
    """
    dataset = AutonomousDrivingDatasetSeg(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get('tubelet_size', 2),
        camera_frame=kwargs.get('camera_frame', False),
        load_segmentation=load_segmentation,  # 新增
        seg_data_root=seg_data_root,  # 新增
        crop_size = crop_size
    )
    
    print(f"创建数据集: {len(dataset)} 个样本")
    
    # 使用支持分割标注的collate函数
    if collator is None:
        collator = autonomous_driving_collate_fn_with_seg
    
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # 数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    
    print(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")
    
    return loader, sampler

class AutonomousDrivingDatasetSeg(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 兼容V-JEPA训练 + 分割标注"""
    
    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
        load_segmentation=True,  # 新增：是否加载分割标注
        seg_data_root="/disk/deepdata/dataset/nvs/data/sam3_autolabeling_point",  # 新增：分割数据根目录
        crop_size = 256,
        num_seg_sample = 4
    ):
        """
        Args:
            data_path: train.txt文件路径
            camera_views: 要使用的相机列表
            frames_per_clip: 每个clip的帧数
            fps: 目标帧率
            transform: 数据增强
            camera_frame: 是否转换到相机坐标系
            frameskip: 帧间隔（tubelet_size）
            load_segmentation: 是否加载分割标注
            seg_data_root: 分割标注数据根目录
        """
        self.data_path = data_path
        self.camera_views = camera_views
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        self.load_segmentation = load_segmentation
        self.seg_data_root = seg_data_root
        self.crop_size = crop_size
        self.num_seg_sample = num_seg_sample
        # 加载样本列表
        with open(data_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        print(f"加载了 {len(self.samples)} 个训练样本")
        if self.load_segmentation:
            print(f"将从 {self.seg_data_root} 加载分割标注")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        
        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"加载失败 {path} (retry {retry+1}/{max_retries}): {e}")
                    index = np.random.randint(len(self))
                    print(f"error is {e}")
                    path = self.samples[index]
                else:
                    raise e

    def load_segmentation_masks_robust(self, trajectory_name, camera_view, indices, 
                                   min_area=50, 
                                   color_tolerance=5,
                                   use_morphology=False,
                                   use_connected_components=True,
                                   visualize=True,
                                   save_vis_path=None):
        """鲁棒的mask加载方法（带可视化）"""
        
        mask_video_path = os.path.join(
            self.seg_data_root, trajectory_name, camera_view,
            f"{camera_view}_pure_mask.mp4"
        )
        
        if not os.path.exists(mask_video_path):
            print(f"警告：分割标注不存在: {mask_video_path}")
            return None
        
        try:
            # 读取视频
            target_index = indices[0]
            vr = VideoReader(mask_video_path, num_threads=0)
            # vlen = len(vr)
            # valid_indices = np.clip(indices, 0, vlen - 1)
            if target_index >= len(vr):
                target_index = len(vr) -1 
            
            vr.seek(0)
            mask_frames = vr.get_batch([target_index]).asnumpy()  # [T, H, W, C]
            mask_tensor = torch.from_numpy(mask_frames).permute(0, 3, 1, 2).float()
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor,
                size=(self.crop_size, self.crop_size),
                mode="nearest",
            )
            mask_frames = mask_resized.permute(0, 2, 3, 1).byte().numpy()
            T, H, W, C = mask_frames.shape
            # 判断是否为黑色：所有通道都为0
            is_black = np.all(mask_frames == 0, axis=-1, keepdims=True)  # [T, H, W, 1]
            
            # 创建二值mask：黑色=255，有色=1
            binary_mask = np.where(is_black, 255, 1).astype(np.uint8)  # [T, H, W, 1]
            
            # 扩展为单通道mask格式 [T, 1, H, W] 以兼容后续可能的处理
            binary_mask = binary_mask.squeeze(-1)  # [T, H, W]
            masks_np = binary_mask[:, np.newaxis, :, :]  # [T, 1, H, W]
            breakpoint()
            # 如果需要可视化
            if visualize:
                # 创建可视化用的彩色图（可选：将255显示为白色，1显示为红色）
                vis_frames = np.repeat(binary_mask[:, :, :, np.newaxis], 3, axis=-1)  # [T, H, W, 3]
                # 255 -> 白色 [255,255,255], 1 -> 红色 [255,0,0]
                vis_frames[binary_mask == 255] = [255, 255, 255]
                vis_frames[binary_mask == 1] = [255, 0, 0]
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_frames[0])  # 原始mask
                plt.title("Original Mask")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(vis_frames[0])  # 二值化结果
                plt.title("Binary Mask (255=black, 1=color)")
                plt.axis('off')
                
                if save_vis_path:
                    plt.savefig(save_vis_path, bbox_inches='tight', dpi=150)
                    print(f"可视化结果已保存: {save_vis_path}")
                plt.show()
            
            return torch.from_numpy(masks_np)
            
        except Exception as e:
            print(f"加载分割标注失败 {mask_video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
        #     start_time = time.time()
        #     # 步骤1：颜色量化
        #     # mask_frames_quantized = (mask_frames // color_tolerance) * color_tolerance
        #     mask_frames_quantized = mask_frames
        #     rgb_ids = (mask_frames_quantized[..., 0].astype(np.int32) * 65536 + 
        #             mask_frames_quantized[..., 1].astype(np.int32) * 256 + 
        #             mask_frames_quantized[..., 2].astype(np.int32))
        #     rgb_ids_flat = rgb_ids.ravel()
        #     max_id = rgb_ids_flat.max()
            
        #     if max_id >= 2**31:
        #         print(f"警告：颜色ID过大")
        #         return None
            
        #     counts = np.bincount(rgb_ids_flat, minlength=max_id + 1)
        #     valid_mask = (counts >= min_area) & (np.arange(len(counts)) != 0)
        #     valid_ids = np.where(valid_mask)[0]
            
        #     if len(valid_ids) == 0:
        #         print(f"警告：过滤后未检测到任何物体mask (min_area={min_area})")
        #         return None
            
        #     N = len(valid_ids)
        #     # print(f"步骤1 - 颜色量化+面积过滤: {np.sum(counts > 0) - 1} → {N} 个物体")
        #     # breakpoint()
        #     # if N > 100:
        #     #     valid_ids = valid_ids[:,:100]
        #     #     N = 100
        #     # 创建初始masks
        #     if N < 1000:
        #         masks_np = (rgb_ids[:, None, :, :] == valid_ids[None, :, None, None]).astype(np.uint8)
        #     else:
        #         masks_np = np.zeros((T, N, H, W), dtype=np.uint8)
        #         for i, color_id in enumerate(valid_ids):
        #             masks_np[:, i, :, :] = (rgb_ids == color_id).astype(np.uint8)
        #     # 步骤2：形态学清理
        #     if use_morphology:
        #         from scipy import ndimage
        #         kernel = np.ones((3, 3), dtype=bool)
                
        #         print("步骤2 - 形态学清理...")
        #         for t in range(T):
        #             for n in range(N):
        #                 mask = masks_np[t, n] > 0.5
        #                 mask = ndimage.binary_closing(mask, structure=kernel)
        #                 mask = ndimage.binary_opening(mask, structure=kernel)
        #                 masks_np[t, n] = mask.astype(np.float32)
        #     # ✅ 关键修复：步骤3 - 连通域过滤（同步更新valid_ids）
        #     if use_connected_components:
        #         try:
        #             from scipy import ndimage
                    
        #             # print("步骤3 - 连通域过滤...")
        #             refined_masks = []
        #             refined_valid_ids = []  # ✅ 新增：同步跟踪有效的颜色ID
        #             removed_count = 0
                    
        #             for n in range(N):
        #                 mask_3d = masks_np[:, n, :, :] > 0
        #                 labeled, num_features = ndimage.label(mask_3d)
                        
        #                 if num_features > 0:
        #                     component_sizes = np.bincount(labeled.ravel())[1:]
        #                     largest = np.argmax(component_sizes) + 1
        #                     refined_mask = (labeled == largest).astype(np.float32)
                            
        #                     if refined_mask.sum() >= min_area:
        #                         refined_masks.append(refined_mask)
        #                         refined_valid_ids.append(valid_ids[n])  # ✅ 保留对应的颜色ID
        #                     else:
        #                         removed_count += 1
        #                 else:
        #                     removed_count += 1
        #             end_time = time.time()
        #             process_time = end_time -start_time
        #             # print(f"seg mask process time is {process_time}")
        #             if len(refined_masks) > 0:
        #                 masks_np = np.stack(refined_masks, axis=1)
        #                 valid_ids = np.array(refined_valid_ids)  # ✅ 更新valid_ids
        #                 # print(f"连通域过滤: {N} → {masks_np.shape[1]} 个物体 (移除 {removed_count} 个)")
        #             else:
        #                 print("警告：连通域过滤后无物体保留")
        #                 return None
        #         except MemoryError:
        #             print(f"MemoryError during morphology/labeling for {trajectory_name}")
        #             return None
        #     final_N = masks_np.shape[1]
        #     # print(f"\n最终保留 {final_N} 个物体")
        #     # ✅ 验证形状一致性
        #     assert len(valid_ids) == final_N, f"valid_ids长度({len(valid_ids)})与masks数量({final_N})不一致"
        #     save_vis_path = '/disk/deepdata/hch_workspace/code/vjepa2/vis_test_last_tole_first_frame.png'
        #     # 可视化
        #     if visualize:
        #         visualize_masks(
        #             mask_frames=mask_frames,
        #             masks=masks_np,
        #             valid_ids=valid_ids,
        #             counts=counts,
        #             trajectory_name=trajectory_name,
        #             camera_view=camera_view,
        #             save_path=save_vis_path
        #         )
        #     return torch.from_numpy(masks_np)
            
        # except Exception as e:
        #     print(f"加载分割标注失败 {mask_video_path}: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     return None
    def load_clip(self, path):
        """加载单个clip的数据（包含分割标注）"""
        # 提取trajectory名称
        trajectory_name = os.path.basename(path)
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            
            # 加载相机外参
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
        
        # 4. 加载视频
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=0)
        vlen = len(vr)
        
        # 5. 计算采样参数
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.frames_per_clip * fstp
        
        if vlen < nframes:
            # 如果视频太短，使用所有帧
            nframes = vlen
            fstp = max(1, vlen // self.frames_per_clip)
        
        # 6. 随机采样时间窗口
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.frames_per_clip]
        
        # 如果帧数不足，填充最后一帧
        if len(indices) < self.frames_per_clip:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.frames_per_clip - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        # 7. 读取视频帧
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        
        # 8. 应用数据增强
        if self.transform is not None:
            buffer = self.transform(buffer)
        
        # 9. 提取对应的状态和外参
        states = states[indices][::self.frameskip]
        extrinsics = extrinsics[indices][::self.frameskip]
        
        # 10. 计算动作（状态差分）
        actions = self.compute_actions(states)
        # ==================== 新增：加载分割标注 ====================
        seg_masks = None
        if self.load_segmentation:
            seg_masks = self.load_segmentation_masks_robust(trajectory_name, camera_view, indices)
            # seg_masks = self.load_segmentation_masks_optimized(trajectory_name, camera_view, indices)
            
            # # 如果加载成功，应用frameskip采样
            if seg_masks is not None:
                seg_masks = seg_masks[::self.frameskip]  # [T//frameskip, H, W]
        
        return {
            'buffer': buffer,           # [T, H, W, C] 或经过transform后的shape
            'actions': actions,         # [T-1, 7]
            'states': states,           # [T, 7]
            'extrinsics': extrinsics,   # [T, 7]
            'indices': indices,         # [T]
            'seg_masks': seg_masks,     # [T, N, H, W]
        }
    def load_clip_list(self, path):
        """加载单个clip的数据（包含分割标注）"""
        trajectory_name = os.path.basename(path)
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
        
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=0)
        vlen = len(vr)
        
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.frames_per_clip * fstp
        
        if vlen < nframes:
            nframes = vlen
            fstp = max(1, vlen // self.frames_per_clip)
        
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.frames_per_clip]
        
        if len(indices) < self.frames_per_clip:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.frames_per_clip - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        
        if self.transform is not None:
            buffer = self.transform(buffer)
        
        # 9. 提取对应的状态和外参
        states = states[indices][::self.frameskip]
        extrinsics = extrinsics[indices][::self.frameskip]
        
        # 10. 计算动作（状态差分）
        actions = self.compute_actions(states)

        # ==================== 新增：加载分割标注 ====================
        seg_masks_list = []
        seg_indices_lst = []
        if self.load_segmentation:
            clip_len = len(indices)
            valid_steps = list(range(clip_len))
            if len(valid_steps) >= self.num_seg_sample:
                selected_relative_indices = sorted(random.sample(valid_steps,self.num_seg_sample))
            else:
                selected_relative_indices = valid_steps
            target_mask_indices = [indices[i] for i in selected_relative_indices]

            # 3. 只加载这几帧
            # Returns: [K, N, H, W] (K = num_seg_samples)
            sampled_masks = self.load_segmentation_masks_robust(
                trajectory_name, camera_view, target_mask_indices
            )

            if sampled_masks is not None:
                if len(sampled_masks) == len(selected_relative_indices):
                    seg_masks_tensor = sampled_masks
                    # 记录这几帧对应的是 Clip 中的第几个时间步 (0~15)
                    # 这对后续 Loss 切片至关重要
                    seg_indices_tensor = torch.tensor(selected_relative_indices, dtype=torch.long)
                else:
                    print(f"Warning: Mask frame count mismatch. Req: {len(target_mask_indices)}, Got: {len(sampled_masks)}")
                    # 简单处理：截断
                    min_len = min(len(sampled_masks), len(selected_relative_indices))
                    seg_masks_tensor = sampled_masks[:min_len]
                    seg_indices_tensor = torch.tensor(selected_relative_indices[:min_len], dtype=torch.long)

        return {
            'buffer': buffer,
            'actions': actions,
            'states': states,
            'extrinsics': extrinsics,
            'indices': indices,
            'seg_masks': seg_masks_tensor,      # [K, N, H, W] 这里的K很小(如4)
            'seg_frame_indices': seg_indices_tensor # [K]
        }
        #     seg_masks = self.load_segmentation_masks_robust(trajectory_name, camera_view, indices)
        #     # seg_masks = self.load_segmentation_masks_optimized(trajectory_name, camera_view, indices)
            
        #     # # 如果加载成功，应用frameskip采样
        #     if seg_masks is not None:
        #         seg_masks = seg_masks[::self.frameskip]  # [T//frameskip, H, W]
        
        # return {
        #     'buffer': buffer,           # [T, H, W, C] 或经过transform后的shape
        #     'actions': actions,         # [T-1, 7]
        #     'states': states,           # [T, 7]
        #     'extrinsics': extrinsics,   # [T, 7]
        #     'indices': indices,         # [T]
        #     'seg_masks': seg_masks,     # [T, N, H, W]
        # }
    def compute_actions(self, states):
        """
        计算动作序列（状态差分）
        
        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        
        Returns:
            actions: [T-1, 7]
        """
        T = len(states)
        actions = np.zeros((T - 1, 7))
        
        for t in range(T - 1):
            # 位置差分
            xyz_diff = states[t + 1, :3] - states[t, :3]
            
            # 旋转差分
            R1 = Rotation.from_euler('xyz', states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler('xyz', states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler('xyz')
            
            # 使用当前帧的速度
            velocity = states[t, 6]
            
            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])
        
        return actions


# ==================== 修改数据collate函数 ====================
def autonomous_driving_collate_fn_with_only_seg(batch):
    """处理分割标注的collate函数"""
    context_frames = torch.stack([item['buffer'] for item in batch])
    actions = torch.stack([torch.from_numpy(item['actions']) for item in batch])
    states = torch.stack([torch.from_numpy(item['states']) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item['extrinsics']) for item in batch])
    
    # 处理分割标注
    seg_targets = []
    for item in batch:
        seg_mask = item.get('seg_masks', None)  # [T, N, H, W]
        indices = item.get('seg_frame_indices',None)
        if seg_mask is not None:
            seg_targets.append((seg_mask,indices))
        else:
            # 空target
            seg_targets.append(None)
    
    return context_frames, actions, states, extrinsics, seg_targets
    
def init_data_only_seg(
    data_path,
    batch_size,
    frames_per_clip=16,
    fps=5,
    camera_views=["CAM_FRONT"],
    transform=None,
    collator=None,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    rank=0,
    world_size=1,
    load_segmentation=True,  # 新增
    seg_data_root="/disk/deepdata/dataset/nvs/data/sam3_autolabeling_point",  # 新增
    crop_size = 256,
    num_seg_sample = 4,
    **kwargs
):
    """
    初始化自动驾驶数据加载器
    """
    dataset = AutonomousDrivingDatasetOnlySeg(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        fps=fps,
        camera_views=camera_views,
        transform=transform,
        frameskip=kwargs.get('tubelet_size', 2),
        camera_frame=kwargs.get('camera_frame', False),
        load_segmentation=load_segmentation,  # 新增
        seg_data_root=seg_data_root,  # 新增
        crop_size = crop_size,
        num_seg_sample = 4
    )
    
    print(f"创建数据集: {len(dataset)} 个样本")
    
    # 使用支持分割标注的collate函数
    if collator is None:
        collator = autonomous_driving_collate_fn_with_only_seg
    
    # 分布式采样器
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    
    # 数据加载器
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0) and persistent_workers,
        drop_last=True,
    )
    
    print(f"数据加载器创建完成: batch_size={batch_size}, num_workers={num_workers}")
    
    return loader, sampler

class AutonomousDrivingDatasetOnlySeg(torch.utils.data.Dataset):
    """自动驾驶视频数据集 - 兼容V-JEPA训练 + 分割标注"""
    
    def __init__(
        self,
        data_path,
        camera_views=["CAM_FRONT"],
        frames_per_clip=16,
        fps=5,
        transform=None,
        camera_frame=False,
        frameskip=2,
        load_segmentation=True,  # 新增：是否加载分割标注
        seg_data_root="/disk/deepdata/dataset/nvs/data/sam3_autolabeling_point",  # 新增：分割数据根目录
        crop_size = 256,
        num_seg_sample = 4
    ):
        """
        Args:
            data_path: train.txt文件路径
            camera_views: 要使用的相机列表
            frames_per_clip: 每个clip的帧数
            fps: 目标帧率
            transform: 数据增强
            camera_frame: 是否转换到相机坐标系
            frameskip: 帧间隔（tubelet_size）
            load_segmentation: 是否加载分割标注
            seg_data_root: 分割标注数据根目录
        """
        self.data_path = data_path
        self.camera_views = camera_views
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.transform = transform
        self.camera_frame = camera_frame
        self.frameskip = frameskip
        self.load_segmentation = load_segmentation
        self.seg_data_root = seg_data_root
        self.crop_size = crop_size
        self.num_seg_sample = num_seg_sample
        # 加载样本列表
        with open(data_path, 'r') as f:
            self.samples = [line.strip() for line in f.readlines()]
        
        print(f"加载了 {len(self.samples)} 个训练样本")
        if self.load_segmentation:
            print(f"将从 {self.seg_data_root} 加载分割标注")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path = self.samples[index]
        
        max_retries = 5
        for retry in range(max_retries):
            try:
                data = self.load_clip(path)
                return data
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"加载失败 {path} (retry {retry+1}/{max_retries}): {e}")
                    index = np.random.randint(len(self))
                    print(f"error is {e}")
                    path = self.samples[index]
                else:
                    raise e

    def load_segmentation_masks_robust(self, trajectory_name, camera_view, indices, 
                                   min_area=50, 
                                   color_tolerance=10,
                                   use_morphology=False,
                                   use_connected_components=True,
                                   visualize=False,
                                   save_vis_path=None):
        """鲁棒的mask加载方法（带可视化）"""
        
        # mask_video_path = os.path.join(
        #     self.seg_data_root, trajectory_name, camera_view,
        #     f"{camera_view}_pure_mask.mp4"
        # )

        mask_video_path = os.path.join(
            self.seg_data_root, trajectory_name,
            f"mask_video.mp4"
        )
        
        if not os.path.exists(mask_video_path):
            print(f"警告：分割标注不存在: {mask_video_path}")
            return None
        
        try:
            # 读取视频
            # target_index = indices[0]
            vr = VideoReader(mask_video_path, num_threads=0)
            vlen = len(vr)
            safe_indices = [idx for idx in indices if idx < vlen]
            if len(safe_indices) == 0 :
                print(f"warning indices over mask video range{vlen}") 
            vr.seek(0)
            mask_frames = vr.get_batch(safe_indices).asnumpy()  # [T, H, W, C]
            mask_tensor = torch.from_numpy(mask_frames).permute(0, 3, 1, 2).float()
            mask_resized = torch.nn.functional.interpolate(
                mask_tensor,
                size=(self.crop_size, self.crop_size),
                mode="nearest",
            )
            mask_frames = mask_resized.permute(0, 2, 3, 1).byte().numpy()
            T, H, W, C = mask_frames.shape
            # 判断是否为黑色：所有通道都为0
            # is_black = np.all(mask_frames == 0, axis=-1, keepdims=True)  # [T, H, W, 1]
            is_black = np.all(mask_frames <= color_tolerance, axis=-1,keepdims=True)
            # 创建二值mask：黑色=255，有色=1
            binary_mask = np.where(is_black, 255, 1).astype(np.uint8)  # [T, H, W, 1]
            
            # 扩展为单通道mask格式 [T, 1, H, W] 以兼容后续可能的处理
            binary_mask = binary_mask.squeeze(-1)  # [T, H, W]
            masks_np = binary_mask[:, np.newaxis, :, :]  # [T, 1, H, W]
            # save_vis_path = '/disk/deepdata/hch_workspace/code/vjepa2/results/vis/mask_new_data.png'
            # 如果需要可视化
            if visualize:
                # 创建可视化用的彩色图（可选：将255显示为白色，1显示为红色）
                vis_frames = np.repeat(binary_mask[:, :, :, np.newaxis], 3, axis=-1)  # [T, H, W, 3]
                # 255 -> 白色 [255,255,255], 1 -> 红色 [255,0,0]
                vis_frames[binary_mask == 255] = [255, 255, 255]
                vis_frames[binary_mask == 1] = [255, 0, 0]
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(mask_frames[0])  # 原始mask
                plt.title("Original Mask")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(vis_frames[0])  # 二值化结果
                plt.title("Binary Mask (255=black, 1=color)")
                plt.axis('off')
                
                if save_vis_path:
                    plt.savefig(save_vis_path, bbox_inches='tight', dpi=150)
                    print(f"可视化结果已保存: {save_vis_path}")
                plt.show()
            return torch.from_numpy(masks_np)
            
        except Exception as e:
            print(f"加载分割标注失败 {mask_video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
            T, H, W, C = mask_frames.shape
            start_time = time.time()
            # 步骤1：颜色量化
            # mask_frames_quantized = (mask_frames // color_tolerance) * color_tolerance
            mask_frames_quantized = mask_frames
            rgb_ids = (mask_frames_quantized[..., 0].astype(np.int32) * 65536 + 
                    mask_frames_quantized[..., 1].astype(np.int32) * 256 + 
                    mask_frames_quantized[..., 2].astype(np.int32))
            rgb_ids_flat = rgb_ids.ravel()
            max_id = rgb_ids_flat.max()
            
            if max_id >= 2**31:
                print(f"警告：颜色ID过大")
                return None
            
            counts = np.bincount(rgb_ids_flat, minlength=max_id + 1)
            valid_mask = (counts >= min_area) & (np.arange(len(counts)) != 0)
            valid_ids = np.where(valid_mask)[0]
            
            if len(valid_ids) == 0:
                print(f"警告：过滤后未检测到任何物体mask (min_area={min_area})")
                return None
            
            N = len(valid_ids)
            # print(f"步骤1 - 颜色量化+面积过滤: {np.sum(counts > 0) - 1} → {N} 个物体")
            # breakpoint()
            # if N > 100:
            #     valid_ids = valid_ids[:,:100]
            #     N = 100
            # 创建初始masks
            if N < 1000:
                masks_np = (rgb_ids[:, None, :, :] == valid_ids[None, :, None, None]).astype(np.uint8)
            else:
                masks_np = np.zeros((T, N, H, W), dtype=np.uint8)
                for i, color_id in enumerate(valid_ids):
                    masks_np[:, i, :, :] = (rgb_ids == color_id).astype(np.uint8)
            # 步骤2：形态学清理
            if use_morphology:
                from scipy import ndimage
                kernel = np.ones((3, 3), dtype=bool)
                
                print("步骤2 - 形态学清理...")
                for t in range(T):
                    for n in range(N):
                        mask = masks_np[t, n] > 0.5
                        mask = ndimage.binary_closing(mask, structure=kernel)
                        mask = ndimage.binary_opening(mask, structure=kernel)
                        masks_np[t, n] = mask.astype(np.float32)
            # ✅ 关键修复：步骤3 - 连通域过滤（同步更新valid_ids）
            if use_connected_components:
                try:
                    from scipy import ndimage
                    
                    # print("步骤3 - 连通域过滤...")
                    refined_masks = []
                    refined_valid_ids = []  # ✅ 新增：同步跟踪有效的颜色ID
                    removed_count = 0
                    
                    for n in range(N):
                        mask_3d = masks_np[:, n, :, :] > 0
                        labeled, num_features = ndimage.label(mask_3d)
                        
                        if num_features > 0:
                            component_sizes = np.bincount(labeled.ravel())[1:]
                            largest = np.argmax(component_sizes) + 1
                            refined_mask = (labeled == largest).astype(np.float32)
                            
                            if refined_mask.sum() >= min_area:
                                refined_masks.append(refined_mask)
                                refined_valid_ids.append(valid_ids[n])  # ✅ 保留对应的颜色ID
                            else:
                                removed_count += 1
                        else:
                            removed_count += 1
                    end_time = time.time()
                    process_time = end_time -start_time
                    # print(f"seg mask process time is {process_time}")
                    if len(refined_masks) > 0:
                        masks_np = np.stack(refined_masks, axis=1)
                        valid_ids = np.array(refined_valid_ids)  # ✅ 更新valid_ids
                        # print(f"连通域过滤: {N} → {masks_np.shape[1]} 个物体 (移除 {removed_count} 个)")
                    else:
                        print("警告：连通域过滤后无物体保留")
                        return None
                except MemoryError:
                    print(f"MemoryError during morphology/labeling for {trajectory_name}")
                    return None
            final_N = masks_np.shape[1]
            # print(f"\n最终保留 {final_N} 个物体")
            # ✅ 验证形状一致性
            assert len(valid_ids) == final_N, f"valid_ids长度({len(valid_ids)})与masks数量({final_N})不一致"
            save_vis_path = '/disk/deepdata/hch_workspace/code/vjepa2/vis_test_last_tole_first_frame.png'
            # 可视化
            if visualize:
                visualize_masks(
                    mask_frames=mask_frames,
                    masks=masks_np,
                    valid_ids=valid_ids,
                    counts=counts,
                    trajectory_name=trajectory_name,
                    camera_view=camera_view,
                    save_path=save_vis_path
                )
            return torch.from_numpy(masks_np)
            
        except Exception as e:
            print(f"加载分割标注失败 {mask_video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def load_clip(self, path):
        """加载单个clip的数据（包含分割标注）"""
        trajectory_name = os.path.basename(path)
        # 1. 加载metadata
        with open(os.path.join(path, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        # 2. 随机选择一个可用的相机
        available_cameras = [cam for cam in self.camera_views 
                            if f"{cam.lower()}_mp4_path" in metadata]
        
        if not available_cameras:
            raise ValueError(f"没有可用的相机视角: {self.camera_views}")
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        # 3. 加载trajectory数据
        h5_path = os.path.join(path, "trajectory.h5")
        with h5py.File(h5_path, 'r') as f:
            # 加载状态 [T, 7]: [x, y, z, roll, pitch, yaw, velocity]
            cartesian_pos = f['observation/robot_state/cartesian_position'][:]
            velocity = f['observation/robot_state/gripper_position'][:]
            states = np.concatenate([cartesian_pos, velocity[:, None]], axis=1)
            
            extrinsics = f[f'observation/camera_extrinsics/{camera_view}_left'][:]
                
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(path, metadata[video_key])
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        vr = VideoReader(video_path, num_threads=0)
        vlen = len(vr)
        
        vfps = vr.get_avg_fps()
        fstp = max(1, int(vfps / self.fps))
        nframes = self.frames_per_clip * fstp
        
        if vlen < nframes:
            nframes = vlen
            fstp = max(1, vlen // self.frames_per_clip)
        
        start_frame = np.random.randint(0, max(1, vlen - nframes + 1))
        indices = np.arange(start_frame, min(start_frame + nframes, vlen), fstp)
        indices = indices[:self.frames_per_clip]
        
        if len(indices) < self.frames_per_clip:
            last_index = indices[-1] if len(indices) > 0 else 0
            indices = np.pad(indices, (0, self.frames_per_clip - len(indices)), 
                           mode='constant', constant_values=last_index)
        
        vr.seek(0)
        buffer = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
        
        if self.transform is not None:
            buffer = self.transform(buffer)
        
        # 9. 提取对应的状态和外参
        states = states[indices][::self.frameskip]
        extrinsics = extrinsics[indices][::self.frameskip]
        
        # 10. 计算动作（状态差分）
        actions = self.compute_actions(states)
        
        # ==================== 新增：加载分割标注 ====================
        seg_masks_tensor = None
        seg_indices_tensor = None
        if self.load_segmentation:
            clip_len = len(indices)
            valid_choices = list(range(0,clip_len,self.frameskip))

            # if len(valid_choices) >= self.num_seg_sample:
            #     selected_relative_indices = sorted(random.sample(valid_choices,self.num_seg_sample))
            # else:
            selected_relative_indices = valid_choices
            target_mask_indices = [indices[i] for i in selected_relative_indices]

            # 3. 只加载这几帧
            # Returns: [K, N, H, W] (K = num_seg_samples)
            sampled_masks = self.load_segmentation_masks_robust(
                trajectory_name, camera_view, target_mask_indices
            )

            if sampled_masks is not None:
                if len(sampled_masks) == len(selected_relative_indices):
                    seg_masks_tensor = sampled_masks
                    # 记录这几帧对应的是 Clip 中的第几个时间步 (0~15)
                    # 这对后续 Loss 切片至关重要
                    seg_indices_tensor = torch.tensor(selected_relative_indices, dtype=torch.long)
                else:
                    print(f"Warning: Mask frame count mismatch. Req: {len(target_mask_indices)}, Got: {len(sampled_masks)}")
                    # 简单处理：截断
                    min_len = min(len(sampled_masks), len(selected_relative_indices))
                    seg_masks_tensor = sampled_masks[:min_len]
                    seg_indices_tensor = torch.tensor(selected_relative_indices[:min_len], dtype=torch.long)

        return {
            'buffer': buffer,
            'actions': actions,
            'states': states,
            'extrinsics': extrinsics,
            'indices': indices,
            'seg_masks': seg_masks_tensor,      # [K, N, H, W] 这里的K很小(如4)
            'seg_frame_indices': seg_indices_tensor # [K]
        }
        #     seg_masks = self.load_segmentation_masks_robust(trajectory_name, camera_view, indices)
        #     # seg_masks = self.load_segmentation_masks_optimized(trajectory_name, camera_view, indices)
            
        #     # # 如果加载成功，应用frameskip采样
        #     if seg_masks is not None:
        #         seg_masks = seg_masks[::self.frameskip]  # [T//frameskip, H, W]
        
        # return {
        #     'buffer': buffer,           # [T, H, W, C] 或经过transform后的shape
        #     'actions': actions,         # [T-1, 7]
        #     'states': states,           # [T, 7]
        #     'extrinsics': extrinsics,   # [T, 7]
        #     'indices': indices,         # [T]
        #     'seg_masks': seg_masks,     # [T, N, H, W]
        # }
    def compute_actions(self, states):
        """
        计算动作序列（状态差分）
        
        Args:
            states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        
        Returns:
            actions: [T-1, 7]
        """
        T = len(states)
        actions = np.zeros((T - 1, 7))
        
        for t in range(T - 1):
            # 位置差分
            xyz_diff = states[t + 1, :3] - states[t, :3]
            
            # 旋转差分
            R1 = Rotation.from_euler('xyz', states[t, 3:6]).as_matrix()
            R2 = Rotation.from_euler('xyz', states[t + 1, 3:6]).as_matrix()
            R_diff = R2 @ R1.T
            angle_diff = Rotation.from_matrix(R_diff).as_euler('xyz')
            
            # 使用当前帧的速度
            velocity = states[t, 6]
            
            actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])
        
        return actions
    
# ==================== 自动驾驶数据集 + MaskCollator ====================
import json
import math
import os
import pathlib
import warnings
from logging import getLogger
from multiprocessing import Value

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu

from src.datasets.utils.dataloader import ConcatIndices, MonitoredDataset, NondeterministicDataLoader
from src.datasets.utils.weighted_sampler import DistributedWeightedSampler

_GLOBAL_SEED = 0
logger = getLogger()


# ==================== MaskCollator ====================
class MaskCollator(object):
    """
    用于生成 V-JEPA 训练所需的 encoder 和 predictor masks
    
    核心功能：
    1. 为每个样本生成随机的时空 mask block
    2. encoder_mask：模型看到的上下文区域
    3. predictor_mask：模型需要预测的目标区域
    """

    def __init__(
        self,
        cfgs_mask,
        dataset_fpcs,
        crop_size=(224, 224),
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        """
        Args:
            cfgs_mask: List[Dict]，mask 配置列表，每个配置包含：
                - spatial_scale: (min, max) 空间 mask 比例
                - temporal_scale: (min, max) 时间 mask 比例
                - aspect_ratio: (min, max) mask 长宽比
                - num_blocks: 预测目标的数量（npred）
                - max_temporal_keep: 最大上下文帧比例
                - max_keep: 最大保留的 patch 数量
            dataset_fpcs: List[int]，不同数据集的 frames_per_clip
            crop_size: 输入视频的空间尺寸
            patch_size: patch 大小（用于 ViT）
            tubelet_size: 时间维度的 patch 大小
        """
        super(MaskCollator, self).__init__()

        self.mask_generators = dict()
        for fpc in dataset_fpcs:
            self.mask_generators[fpc] = []
            for m in cfgs_mask:
                mask_generator = _MaskGenerator(
                    crop_size=crop_size,
                    num_frames=fpc,
                    spatial_patch_size=patch_size,
                    temporal_patch_size=tubelet_size,
                    spatial_pred_mask_scale=m.get("spatial_scale"),
                    temporal_pred_mask_scale=m.get("temporal_scale"),
                    aspect_ratio=m.get("aspect_ratio"),
                    npred=m.get("num_blocks"),
                    max_context_frames_ratio=m.get("max_temporal_keep", 1.0),
                    max_keep=m.get("max_keep", None),
                    full_complement=m.get("full_complement", False),
                    pred_full_complement=m.get("pred_full_complement", False),
                    inv_block=m.get("inv_block", False),
                )
                self.mask_generators[fpc].append(mask_generator)

    def step(self):
        """更新全局计数器（用于随机种子）"""
        for fpc in self.mask_generators:
            for mask_generator in self.mask_generators[fpc]:
                mask_generator.step()

    def __call__(self, batch):
        """
        处理 batch 数据并生成 masks
        
        Args:
            batch: List[Tuple]，每个元素是 (buffer, label, clip_indices)
        
        Returns:
            fpc_collations: List[Tuple]，每个元素包含：
                - collated_batch: (video, label, clip_indices)
                - collated_masks_enc: List[Tensor]，encoder masks
                - collated_masks_pred: List[Tensor]，predictor masks
        """
        # 按 frames_per_clip 分组
        filtered_batches = {fpc: [] for fpc in self.mask_generators}
        for sample in batch:
            # sample[-1] 是 clip_indices，sample[-1][-1] 是最后一个 clip 的索引
            fpc = len(sample[-1][-1])
            filtered_batches[fpc].append(sample)

        fpc_collations = []
        for fpc in filtered_batches:
            fpc_batch = filtered_batches[fpc]
            batch_size = len(fpc_batch)
            if batch_size == 0:
                continue
            
            # 使用默认 collate 处理视频数据
            collated_batch = torch.utils.data.default_collate(fpc_batch)
            
            # 为当前 batch 生成 masks
            collated_masks_pred, collated_masks_enc = [], []
            for mask_generator in self.mask_generators[fpc]:
                masks_enc, masks_pred = mask_generator(batch_size)
                collated_masks_enc.append(masks_enc)
                collated_masks_pred.append(masks_pred)
            
            fpc_collations.append((collated_batch, collated_masks_enc, collated_masks_pred))

        return fpc_collations


class _MaskGenerator(object):
    """内部类：生成单个 mask 配置的 masks"""

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        inv_block=False,
        full_complement=False,
        pred_full_complement=False,
    ):
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        
        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )
        self.max_keep = max_keep
        self._itr_counter = Value("i", -1)
        self.inv_block = inv_block

    def step(self):
        """增加迭代计数器"""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        """采样 mask block 的时空尺寸"""
        # 采样时间尺度
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # 采样空间尺度
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # 采样长宽比
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # 计算高度和宽度
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        """采样一个 mask block 的位置"""
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # 限制上下文帧数
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        return mask

    def __call__(self, batch_size):
        """
        为一个 batch 生成 encoder 和 predictor masks
        
        Returns:
            collated_masks_enc: [B, N_enc] encoder mask indices
            collated_masks_pred: [B, N_pred] predictor mask indices
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        # 采样 predictor block 尺寸（使用固定种子保证一致性）
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        
        for _ in range(batch_size):
            empty_context = True
            while empty_context:
                # 生成 encoder mask（上下文区域）
                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                # predictor mask 是 encoder mask 的补集
                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        # 截断到最小长度（保证 batch 中所有样本长度一致）
        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        
        # 可选：使用完全补集
        if self.full_complement:
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        if self.inv_block:
            return collated_masks_pred, collated_masks_enc
        else:
            return collated_masks_enc, collated_masks_pred


# ==================== 数据初始化函数（集成 MaskCollator）====================
def init_data_autonomous_driving(
    batch_size,
    transform=None,
    shared_transform=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    clip_len=None,
    dataset_fpcs=None,
    frame_sample_rate=None,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(1e9),
    datasets_weights=None,
    persistent_workers=False,
    deterministic=True,
    log_dir=None,
    camera_views=["CAM_FRONT"],
    # MaskCollator 参数
    collator=None,
    cfgs_mask=None,
    crop_size=(224, 224),
    patch_size=(16, 16),
    tubelet_size=2,
):
    """
    初始化自动驾驶数据集（带 MaskCollator）
    
    Args:
        cfgs_mask: List[Dict]，mask 配置，例如：
            [
                {
                    'spatial_scale': (0.2, 0.8),
                    'temporal_scale': (1.0, 1.0),
                    'aspect_ratio': (0.3, 3.0),
                    'num_blocks': 4,
                    'max_temporal_keep': 1.0,
                    'max_keep': None,
                }
            ]
    """
    # 创建 MaskCollator
    # if cfgs_mask is not None:
    #     if dataset_fpcs is None:
    #         dataset_fpcs = [clip_len]
        
    #     collator = MaskCollator(
    #         cfgs_mask=cfgs_mask,
    #         dataset_fpcs=dataset_fpcs,
    #         crop_size=crop_size,
    #         patch_size=patch_size,
    #         tubelet_size=tubelet_size,
    #     )
    #     logger.info("MaskCollator created")
    # else:
    #     collator = None
    #     logger.warning("No cfgs_mask provided, using default collate")

    dataset, data_loader, dist_sampler = make_autonomous_driving_dataset(
        data_paths=root_path,
        batch_size=batch_size,
        frames_per_clip=clip_len,
        dataset_fpcs=dataset_fpcs,
        frame_step=frame_sample_rate,
        duration=duration,
        fps=fps,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        datasets_weights=datasets_weights,
        collator=collator,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        world_size=world_size,
        rank=rank,
        deterministic=deterministic,
        log_dir=log_dir,
        camera_views=camera_views,
    )
    return data_loader, dist_sampler


def make_autonomous_driving_dataset(
    data_paths,
    batch_size,
    frames_per_clip=8,
    dataset_fpcs=None,
    frame_step=4,
    duration=None,
    fps=None,
    num_clips=1,
    random_clip_sampling=True,
    allow_clip_overlap=False,
    filter_short_videos=False,
    filter_long_videos=int(10**9),
    transform=None,
    shared_transform=None,
    rank=0,
    world_size=1,
    datasets_weights=None,
    collator=None,
    drop_last=True,
    num_workers=10,
    pin_mem=True,
    persistent_workers=True,
    deterministic=True,
    log_dir=None,
    camera_views=["CAM_FRONT"],
):
    """创建数据集和数据加载器"""
    
    dataset = AutonomousDrivingVideoDataset(
        data_paths=data_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        dataset_fpcs=dataset_fpcs,
        duration=duration,
        fps=fps,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        shared_transform=shared_transform,
        transform=transform,
        camera_views=camera_views,
    )

    log_dir = pathlib.Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        resource_log_filename = log_dir / f"resource_file_{rank}_%w.csv"
        dataset = MonitoredDataset(
            dataset=dataset,
            log_filename=str(resource_log_filename),
            log_interval=10.0,
            monitor_interval=5.0,
        )

    logger.info("AutonomousDrivingVideoDataset created")
    
    if datasets_weights is not None:
        dist_sampler = DistributedWeightedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    else:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

    if deterministic:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    else:
        data_loader = NondeterministicDataLoader(
            dataset,
            collate_fn=collator,
            sampler=dist_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_mem,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0) and persistent_workers,
        )
    
    logger.info("AutonomousDrivingVideoDataset data loader created")
    return dataset, data_loader, dist_sampler


# ==================== 数据集类（保持不变）====================
class AutonomousDrivingVideoDataset(torch.utils.data.Dataset):
    """自动驾驶视频数据集"""

    def __init__(
        self,
        data_paths,
        datasets_weights=None,
        frames_per_clip=16,
        fps=None,
        dataset_fpcs=None,
        frame_step=4,
        num_clips=1,
        transform=None,
        shared_transform=None,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=None,
        camera_views=["CAM_FRONT"],
    ):
        self.data_paths = data_paths
        self.datasets_weights = datasets_weights
        self.frame_step = frame_step
        self.num_clips = num_clips
        self.transform = transform
        self.shared_transform = shared_transform
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos
        self.filter_long_videos = filter_long_videos
        self.duration = duration
        self.fps = fps
        self.camera_views = camera_views

        if sum([v is not None for v in (fps, duration, frame_step)]) != 1:
            raise ValueError(f"Must specify exactly one of {fps=}, {duration=}, or {frame_step=}.")

        if isinstance(data_paths, str):
            data_paths = [data_paths]

        if dataset_fpcs is None:
            self.dataset_fpcs = [frames_per_clip for _ in data_paths]
        else:
            if len(dataset_fpcs) != len(data_paths):
                raise ValueError("Frames per clip not properly specified")
            self.dataset_fpcs = dataset_fpcs

        if VideoReader is None:
            raise ImportError('Unable to import "decord"')

        # 加载样本列表
        samples, labels = [], []
        self.num_samples_per_dataset = []
        for data_path in self.data_paths:
            if data_path.endswith(".csv"):
                try:
                    data = pd.read_csv(data_path, header=None, delimiter=" ")
                except pd.errors.ParserError:
                    data = pd.read_csv(data_path, header=None, delimiter="::")
                samples += list(data.values[:, 0])
                labels += list(data.values[:, 1]) if data.shape[1] > 1 else [0] * len(data)
                num_samples = len(data)
                
            elif data_path.endswith(".npy"):
                data = np.load(data_path, allow_pickle=True)
                data = list(map(lambda x: repr(x)[1:-1], data))
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
                
            elif data_path.endswith(".txt"):
                with open(data_path, 'r') as f:
                    data = [line.strip() for line in f.readlines()]
                samples += data
                labels += [0] * len(data)
                num_samples = len(data)
            
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            self.num_samples_per_dataset.append(num_samples)

        self.per_dataset_indices = ConcatIndices(self.num_samples_per_dataset)

        self.sample_weights = None
        if self.datasets_weights is not None:
            self.sample_weights = []
            for dw, ns in zip(self.datasets_weights, self.num_samples_per_dataset):
                self.sample_weights += [dw / ns] * ns

        self.samples = samples
        self.labels = labels
        
        logger.info(f"Loaded {len(self.samples)} autonomous driving samples")

    def __getitem__(self, index):
        sample = self.samples[index]
        loaded_sample = False
        
        max_retries = 5
        retry_count = 0
        while not loaded_sample and retry_count < max_retries:
            loaded_sample = self.get_item_video(index)
            
            if not loaded_sample:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Failed to load {sample}, retry {retry_count}/{max_retries}")
                    index = np.random.randint(self.__len__())
                    sample = self.samples[index]
                else:
                    raise RuntimeError(f"Failed to load sample after {max_retries} retries")

        return loaded_sample

    def get_item_video(self, index):
        """加载单个视频样本"""
        sample = self.samples[index]
        dataset_idx, _ = self.per_dataset_indices[index]
        frames_per_clip = self.dataset_fpcs[dataset_idx]
        buffer, clip_indices = self.loadvideo_decord(sample, frames_per_clip) #(8, 1080, 1920, 3) (T,H,W,C),
        #[array([61, 65, 70, 74, 79, 83, 88, 92])]
        if len(buffer) == 0:
            return None

        label = self.labels[index]

        if self.shared_transform is not None:
            buffer = self.shared_transform(buffer)

        def split_into_clips(video):
            fpc = frames_per_clip
            nc = self.num_clips
            return [video[i * fpc : (i + 1) * fpc] for i in range(nc)]

        buffer = split_into_clips(buffer)
        
        if self.transform is not None:
            buffer = [self.transform(clip) for clip in buffer]
        #torch.Size([3, 8, 256, 256])
        return buffer, label, clip_indices

    def loadvideo_decord(self, sample, fpc):
        """加载视频帧（保持原有逻辑）"""
        fname = sample
        
        if not os.path.exists(fname):
            warnings.warn(f"Sample path not found: {fname}")
            return [], None

        metadata_path = os.path.join(fname, "metadata.json")
        if not os.path.exists(metadata_path):
            warnings.warn(f"Metadata not found: {metadata_path}")
            return [], None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        available_cameras = [
            cam for cam in self.camera_views 
            if f"{cam.lower()}_mp4_path" in metadata
        ]
        
        if not available_cameras:
            warnings.warn(f"No available cameras in {fname}")
            return [], None
        
        camera_view = available_cameras[np.random.randint(len(available_cameras))]
        
        video_key = f"{camera_view.lower()}_mp4_path"
        video_path = os.path.join(fname, metadata[video_key])
        
        if not os.path.exists(video_path):
            warnings.warn(f"Video not found: {video_path}")
            return [], None

        _fsize = os.path.getsize(video_path)
        if _fsize > self.filter_long_videos:
            warnings.warn(f"Skipping long video: {_fsize} bytes")
            return [], None

        try:
            vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        except Exception as e:
            warnings.warn(f"Failed to load video: {e}")
            return [], None

        fstp = self.frame_step
        if self.duration is not None or self.fps is not None:
            try:
                video_fps = math.ceil(vr.get_avg_fps())
            except Exception as e:
                logger.warning(e)
                return [], None

            if self.duration is not None:
                fstp = int(self.duration * video_fps / fpc)
            else:
                fstp = video_fps // self.fps

        assert fstp is not None and fstp > 0
        clip_len = int(fpc * fstp)

        if self.filter_short_videos and len(vr) < clip_len:
            warnings.warn(f"Skipping short video: {len(vr)} frames")
            return [], None

        vr.seek(0)

        partition_len = len(vr) // self.num_clips

        all_indices, clip_indices = [], []
        for i in range(self.num_clips):
            if partition_len > clip_len:
                end_indx = clip_len
                if self.random_clip_sampling:
                    end_indx = np.random.randint(clip_len, partition_len)
                start_indx = end_indx - clip_len
                indices = np.linspace(start_indx, end_indx, num=fpc)
                indices = np.clip(indices, start_indx, end_indx - 1).astype(np.int64)
                indices = indices + i * partition_len
            else:
                if not self.allow_clip_overlap:
                    indices = np.linspace(0, partition_len, num=partition_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - partition_len // fstp) * partition_len,
                    ))
                    indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
                    indices = indices + i * partition_len
                else:
                    sample_len = min(clip_len, len(vr)) - 1
                    indices = np.linspace(0, sample_len, num=sample_len // fstp)
                    indices = np.concatenate((
                        indices,
                        np.ones(fpc - sample_len // fstp) * sample_len,
                    ))
                    indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
                    clip_step = 0
                    if len(vr) > clip_len:
                        clip_step = (len(vr) - clip_len) // (self.num_clips - 1)
                    indices = indices + i * clip_step

            clip_indices.append(indices)
            all_indices.extend(list(indices))

        buffer = vr.get_batch(all_indices).asnumpy()
        return buffer, clip_indices

    def __len__(self):
        return len(self.samples)
