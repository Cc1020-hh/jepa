# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""
轨迹预测评估模块
评估指标:
- ADE (Average Displacement Error): 所有预测点与GT点的平均L2距离
- FDE (Final Displacement Error): 最后一个预测点与GT点的L2距离
"""

import os
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from src.utils.logging import get_logger
from src.utils.distributed import init_distributed
import torch.distributed as dist

logger = get_logger(__name__)


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Average Displacement Error (ADE)

    Args:
        pred_traj: 预测轨迹 [B, num_poses, 3] (x, y, yaw)
        gt_traj: 真实轨迹 [B, num_poses, 3] (x, y, yaw)

    Returns:
        ade: 平均位移误差 [B] 或 scalar
    """
    # 只计算 x, y 的 L2 距离
    pred_xy = pred_traj[..., :2]  # [B, num_poses, 2]
    gt_xy = gt_traj[..., :2]  # [B, num_poses, 2]

    # 每个点的 L2 距离
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, num_poses]

    # 对所有轨迹点取平均
    ade = displacement.mean(dim=-1)  # [B]

    return ade


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """
    计算 Final Displacement Error (FDE)

    Args:
        pred_traj: 预测轨迹 [B, num_poses, 3] (x, y, yaw)
        gt_traj: 真实轨迹 [B, num_poses, 3] (x, y, yaw)

    Returns:
        fde: 最终位移误差 [B]
    """
    # 只计算 x, y 的 L2 距离，取最后一个点
    pred_xy_final = pred_traj[:, -1, :2]  # [B, 2]
    gt_xy_final = gt_traj[:, -1, :2]  # [B, 2]

    # 最后一个点的 L2 距离
    fde = torch.norm(pred_xy_final - gt_xy_final, dim=-1)  # [B]

    return fde


def compute_metrics(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    """
    计算所有评估指标

    Args:
        pred_traj: 预测轨迹 [B, num_poses, 3]
        gt_traj: 真实轨迹 [B, num_poses, 3]

    Returns:
        metrics: 包含 ADE, FDE 的字典
    """
    ade_per_sample = compute_ade(pred_traj, gt_traj)  # [B]
    fde_per_sample = compute_fde(pred_traj, gt_traj)  # [B]

    return {
        'ade': ade_per_sample.mean().item(),  # scalar
        'fde': fde_per_sample.mean().item(),  # scalar
        'ade_per_sample': ade_per_sample,  # [B]
        'fde_per_sample': fde_per_sample,  # [B]
    }


def prepare_status_feature(states, actions):
    """从 states 和 actions 提取状态特征 (与 train.py 保持一致)"""
    B = states.shape[0]
    velocity = states[:, -1, 6:7]  # [B, 1]
    if states.shape[1] >= 2:
        acceleration = states[:, -1, 6:7] - states[:, -2, 6:7]
    else:
        acceleration = torch.zeros_like(velocity)
    yaw = states[:, -1, 5:6]  # [B, 1]
    xy = states[:, -1, 0:2]  # [B, 2]
    if actions is not None and actions.shape[1] > 0:
        last_action = actions[:, -1, :3]
    else:
        last_action = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    return torch.cat([velocity, acceleration, yaw, xy, last_action], dim=-1)  # [B, 8]


@torch.no_grad()
def validate_one_epoch(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    device,
    dtype,
    mixed_precision,
    tubelet_size,
    tokens_per_frame,
    num_poses,
    world_size,
    rank,
    epoch,
    use_tubelet_repeat: bool = False,
) -> dict:
    """
    执行一个完整epoch的验证

    Args:
        encoder: 编码器模型
        predictor: 预测器模型
        planner: 轨迹规划器
        val_loader: 验证数据加载器
        val_sampler: 验证数据采样器
        device: 设备
        dtype: 数据类型
        mixed_precision: 是否使用混合精度
        tubelet_size: 时间压缩率
        tokens_per_frame: 每帧token数
        num_poses: 轨迹点数
        world_size: 分布式训练的world size
        rank: 当前进程rank
        epoch: 当前epoch
        use_tubelet_repeat: 是否对采样帧执行tubelet_size复制操作

    Returns:
        metrics: 包含平均 ADE, FDE 的字典
    """
    # 设置模型为评估模式
    encoder.eval()
    predictor.eval()
    planner.eval()

    # 累计指标
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    # 设置采样器
    val_sampler.set_epoch(epoch)

    for batch_idx, sample in enumerate(val_loader):
        try:
            # 加载数据
            context_frames = sample[0].to(device, non_blocking=True)  # [B, C, T, H, W]
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B, 15, 7]
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B, 16, 7]
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)  # [B, 16, 7]

            B = context_frames.shape[0]

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # 根据 use_tubelet_repeat 参数决定是否执行帧采样和复制
                if use_tubelet_repeat:
                    # 只取 ::tubelet_size 的帧
                    sampled_frames = context_frames[:, :, ::tubelet_size, :, :]  # [B, C, T//tubelet_size, H, W]
                    C = sampled_frames.shape[1]
                    H, W = sampled_frames.shape[3], sampled_frames.shape[4]

                    # 对每帧复制 tubelet_size 次扩展成视频格式
                    sampled_frames = sampled_frames.unsqueeze(3).repeat(1, 1, 1, tubelet_size, 1, 1)
                    sampled_frames = sampled_frames.view(B, C, -1, H, W)

                    # Encoder forward
                    z_context = encoder(sampled_frames.permute(0, 2, 1, 3, 4))
                else:
                    # 直接使用原始帧
                    z_context = encoder(context_frames.permute(0, 2, 1, 3, 4))
                z = z_context.last_hidden_state  # [B, N, D]

                # Predictor forward (autoregressive)
                _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
                z_tf = predictor(_z, _a, _s, _e)

                # Autoregressive rollout
                _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
                num_prediction_steps = z.size()[1] // tokens_per_frame - 1

                for k in range(1, num_prediction_steps):
                    if k == num_prediction_steps - 1:
                        _a, _s, _e = actions, states[:, :-1], extrinsics[:, :-1]
                    else:
                        _a, _s, _e = actions[:, :k+1], states[:, :k+1], extrinsics[:, :k+1]
                    _z_nxt = predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                    _z = torch.cat([_z, _z_nxt], dim=1)

                z_ar = _z[:, tokens_per_frame:]

                # Planner forward
                status_feature = prepare_status_feature(states, actions)
                planner_output = planner(z_ar, status_feature)

                # 支持多模态 planner (train_mm.py) 和单模态 planner (train.py)
                if "trajectories" in planner_output:
                    # 多模态输出：选择置信度最高的轨迹
                    pred_trajs = planner_output["trajectories"]   # [B, K, num_poses, 3]
                    pred_conf = planner_output["confidences"]     # [B, K]
                    best_idx = pred_conf.argmax(dim=1)            # [B]
                    # 扩展索引以 gather
                    best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(-1, 1, pred_trajs.shape[2], pred_trajs.shape[3])
                    traj_output = pred_trajs.gather(1, best_idx_exp).squeeze(1)  # [B, num_poses, 3]
                else:
                    # 单模态输出
                    traj_output = planner_output["trajectory"]  # [B, num_poses, 3]

                # GT 轨迹转换（世界坐标 -> ego相对坐标）
                StateSE2_indices = [0, 1, 5]
                states_se2 = states[:, :, StateSE2_indices]  # [B, T, 3]

                origin_x = states_se2[:, 0, 0]
                origin_y = states_se2[:, 0, 1]
                origin_yaw = states_se2[:, 0, 2]

                # 相对平移
                dx = states_se2[:, 1:, 0] - origin_x[:, None]
                dy = states_se2[:, 1:, 1] - origin_y[:, None]
                dyaw = states_se2[:, 1:, 2] - origin_yaw[:, None]

                # 旋转到ego坐标系
                cos_h = torch.cos(-origin_yaw)
                sin_h = torch.sin(-origin_yaw)
                ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                # 取前 num_poses 个点
                gt_trajectory = torch.stack([
                    ego_x[:, :num_poses],
                    ego_y[:, :num_poses],
                    ego_yaw[:, :num_poses]
                ], dim=-1)  # [B, num_poses, 3]

                # 计算指标
                metrics = compute_metrics(traj_output, gt_trajectory)
                batch_ade = metrics['ade']
                batch_fde = metrics['fde']

                total_ade += batch_ade * B
                total_fde += batch_fde * B
                total_samples += B

            if batch_idx % 50 == 0:
                logger.info(f"Validation Epoch {epoch}, Batch {batch_idx}/{len(val_loader)}, "
                           f"ADE: {batch_ade:.4f}, FDE: {batch_fde:.4f}")

        except Exception as e:
            logger.warning(f"Validation batch {batch_idx} failed: {e}")
            continue

    # 计算平均指标
    avg_ade = total_ade / total_samples if total_samples > 0 else 0.0
    avg_fde = total_fde / total_samples if total_samples > 0 else 0.0

    # 分布式同步: 聚合所有进程的指标
    if world_size > 1:
        # 转换为tensor便于同步
        metrics_tensor = torch.tensor([avg_ade, avg_fde, total_samples],
                                       dtype=torch.float32, device=device)

        # All-reduce 聚合
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        # 重新计算平均值
        total_samples_all = metrics_tensor[2].item()
        if total_samples_all > 0:
            avg_ade = metrics_tensor[0].item() / total_samples_all
            avg_fde = metrics_tensor[1].item() / total_samples_all

    # 恢复训练模式
    encoder.train()
    predictor.train()
    planner.train()

    return {
        'ade': avg_ade,
        'fde': avg_fde,
    }


def run_validation(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    config: dict,
    epoch: int,
    rank: int,
    world_size: int,
    use_tubelet_repeat: bool = False,
) -> dict:
    """
    运行验证的入口函数

    Args:
        encoder: 编码器 (可能是DDP包装)
        predictor: 预测器 (可能是DDP包装)
        planner: 规划器 (可能是DDP包装)
        val_loader: 验证数据加载器
        val_sampler: 验证采样器
        config: 配置字典
        epoch: 当前epoch
        rank: 当前进程rank
        world_size: world size
        use_tubelet_repeat: 是否对采样帧执行tubelet_size复制操作

    Returns:
        metrics: 验证指标字典
    """
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据类型
    which_dtype = config.get("meta", {}).get("dtype", "float32")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # 数据配置
    cfgs_data = config.get("data", {})
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    target_frame = cfgs_data.get("num_target_frames", 16)
    tokens_per_frame = int((crop_size // patch_size) ** 2)
    num_poses = (target_frame // tubelet_size) - 1

    logger.info(f"Starting validation for epoch {epoch}...")
    logger.info(f"Validation config: tokens_per_frame={tokens_per_frame}, num_poses={num_poses}, use_tubelet_repeat={use_tubelet_repeat}")

    metrics = validate_one_epoch(
        encoder=encoder,
        predictor=predictor,
        planner=planner,
        val_loader=val_loader,
        val_sampler=val_sampler,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        tubelet_size=tubelet_size,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        use_tubelet_repeat=use_tubelet_repeat,
    )

    if rank == 0:
        logger.info(f"="*50)
        logger.info(f"Validation Results - Epoch {epoch}:")
        logger.info(f"  ADE (Average Displacement Error): {metrics['ade']:.4f} m")
        logger.info(f"  FDE (Final Displacement Error):    {metrics['fde']:.4f} m")
        logger.info(f"="*50)

    return metrics
