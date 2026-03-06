# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""
轨迹预测评估模块 (for train_giant.py)
与 train_giant.py 的 encoder/predictor 调用方式保持一致

评估指标:
- ADE (Average Displacement Error): 所有预测点与GT点的平均L2距离
- FDE (Final Displacement Error): 最后一个预测点与GT点的L2距离
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from src.utils.logging import get_logger
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


def compute_minade_minfde_k(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    """
    计算多模态轨迹的 minADE@K / minFDE@K

    Args:
        pred_trajs: 预测轨迹 [B, K, num_poses, 3]
        gt_traj:    真实轨迹 [B, num_poses, 3]

    Returns:
        dict: 包含 minade_k / minfde_k 及逐样本张量
    """
    # [B, K, num_poses, 2]
    pred_xy = pred_trajs[..., :2]
    gt_xy = gt_traj[:, None, :, :2]
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, K, num_poses]

    ade_k = displacement.mean(dim=-1)                   # [B, K]
    fde_k = displacement[:, :, -1]                      # [B, K]

    minade_per_sample = ade_k.min(dim=1).values         # [B]
    minfde_per_sample = fde_k.min(dim=1).values         # [B]

    return {
        "minade_k": minade_per_sample.mean().item(),
        "minfde_k": minfde_per_sample.mean().item(),
        "minade_per_sample": minade_per_sample,
        "minfde_per_sample": minfde_per_sample,
    }


def prepare_status_feature(states, actions, mode: str = "last"):
    """从 states 和 actions 提取状态特征 (与 train_giant.py 保持一致)。"""
    B = states.shape[0]
    if mode == "first":
        idx = 0
        velocity = states[:, idx, 6:7]
        if states.shape[1] >= 2:
            acceleration = states[:, 1, 6:7] - states[:, 0, 6:7]
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, idx, 5:6]
        xy = states[:, idx, 0:2]
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, 0, :3]
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    elif mode == "history_pool":
        velocity = states[:, :, 6:7].mean(dim=1)
        if states.shape[1] >= 2:
            dv = states[:, 1:, 6:7] - states[:, :-1, 6:7]
            acceleration = dv.mean(dim=1)
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, :, 5:6].mean(dim=1)
        xy = states[:, :, 0:2].mean(dim=1)
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, :, :3].mean(dim=1)
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    else:  # "last"
        velocity = states[:, -1, 6:7]
        if states.shape[1] >= 2:
            acceleration = states[:, -1, 6:7] - states[:, -2, 6:7]
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, -1, 5:6]
        xy = states[:, -1, 0:2]
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, -1, :3]
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    return torch.cat([velocity, acceleration, yaw, xy, action_feat], dim=-1)  # [B, 8]


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
    num_time_steps,
    world_size,
    rank,
    epoch,
    normalize_reps: bool = True,
    status_mode: str = "last",
    z_ar_mode: str = "full",
    use_z_context: bool = False,
) -> dict:
    """
    执行一个完整epoch的验证 (与 train_giant.py 逻辑一致)

    Args:
        encoder: 编码器模型 (可能是DDP包装)
        predictor: 预测器模型 (可能是DDP包装)
        planner: 轨迹规划器 (可能是DDP包装)
        val_loader: 验证数据加载器
        val_sampler: 验证数据采样器
        device: 设备
        dtype: 数据类型
        mixed_precision: 是否使用混合精度
        tubelet_size: 时间压缩率
        tokens_per_frame: 每帧token数
        num_poses: 轨迹点数
        num_time_steps: 时间步数
        world_size: 分布式训练的world size
        rank: 当前进程rank
        epoch: 当前epoch
        normalize_reps: 是否对表示进行归一化

    Returns:
        metrics: 包含平均 ADE, FDE 的字典
    """
    # 获取实际模型 (处理DDP包装)
    encoder_unwrapped = encoder.module if hasattr(encoder, 'module') else encoder
    predictor_unwrapped = predictor.module if hasattr(predictor, 'module') else predictor
    planner_unwrapped = planner.module if hasattr(planner, 'module') else planner

    # 设置模型为评估模式
    encoder_unwrapped.eval()
    predictor_unwrapped.eval()
    planner_unwrapped.eval()

    # 累计指标
    total_ade = 0.0
    total_fde = 0.0
    total_minade_k = 0.0
    total_minfde_k = 0.0
    total_samples = 0
    failed_batches = 0

    # 设置采样器
    val_sampler.set_epoch(epoch)

    for batch_idx, sample in enumerate(val_loader):
        try:
            # 加载数据 (与 train_giant.py 保持一致)
            context_frames = sample[0].to(device, non_blocking=True)  # [B, C, T, H, W]
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B, 15, 7]
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)  # [B, 16, 7]
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)  # [B, 16, 7]

            B = context_frames.shape[0]
            C = context_frames.shape[1]
            T = context_frames.shape[2]
            H, W = context_frames.shape[3], context_frames.shape[4]

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # ==================== Encoder Forward (与 train_giant.py forward_context 一致) ====================
                # 只取 ::tubelet_size 的帧
                sampled_clips = context_frames[:, :, ::tubelet_size, :, :]  # [B, C, T//tubelet_size, H, W]
                num_sampled_frames = sampled_clips.shape[2]

                # 对每帧复制 tubelet_size 次扩展成视频格式
                sampled_clips = sampled_clips.unsqueeze(3).repeat(1, 1, 1, tubelet_size, 1, 1)
                sampled_clips = sampled_clips.view(B, C, -1, H, W)  # [B, C, T, H, W]

                # Encoder (vjepa 方式调用: 输入为 list, 返回也为 list)
                z_context = encoder_unwrapped([sampled_clips])
                z = z_context[0]  # [B, N, D]

                if normalize_reps:
                    z = F.layer_norm(z, (z.size(-1),))

                # ==================== Predictor Forward (与 train_giant.py forward_predictions 一致) ====================
                def _step_predictor(_z, _a, _s, _e):
                    _z_out = predictor_unwrapped(_z, _a, _s, _e)
                    if normalize_reps:
                        _z_out = F.layer_norm(_z_out, (_z_out.size(-1),))
                    return _z_out

                # Teacher forcing
                _z, _a, _s, _e = z[:, :-tokens_per_frame], actions, states[:, :-1], extrinsics[:, :-1]
                z_tf = _step_predictor(_z, _a, _s, _e)

                # Autoregressive rollout
                _z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
                num_prediction_steps = z.size()[1] // tokens_per_frame - 1

                for k in range(1, num_prediction_steps):
                    if k == num_prediction_steps - 1:
                        _a, _s, _e = actions, states[:, :-1], extrinsics[:, :-1]
                    else:
                        _a, _s, _e = actions[:, :k+1], states[:, :k+1], extrinsics[:, :k+1]
                    _z_nxt = _step_predictor(_z, _a, _s, _e)[:, -tokens_per_frame:]
                    _z = torch.cat([_z, _z_nxt], dim=1)

                z_ar = _z[:, tokens_per_frame:]
                z_ar_planner = z_ar if z_ar_mode == "full" else z_ar[:, :tokens_per_frame]

                # ==================== Planner Forward ====================
                status_feature = prepare_status_feature(states, actions, mode=status_mode)
                planner_output = planner_unwrapped(
                    z_ar_planner,
                    status_feature,
                    z_context=z if use_z_context else None,
                )

                # 支持多模态 planner
                pred_trajs = None
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

                # ==================== GT 轨迹转换 (与 train_giant.py 一致) ====================
                StateSE2_indices = [0, 1, 5]
                states_se2 = states[:, :, StateSE2_indices]
                origin_x = states_se2[:, 0, 0]
                origin_y = states_se2[:, 0, 1]
                origin_yaw = states_se2[:, 0, 2]

                dx = states_se2[:, 1:, 0] - origin_x[:, None]
                dy = states_se2[:, 1:, 1] - origin_y[:, None]
                dyaw = states_se2[:, 1:, 2] - origin_yaw[:, None]

                cos_h = torch.cos(-origin_yaw)
                sin_h = torch.sin(-origin_yaw)
                ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1)  # [B, T-1, 3]
                gt_trajectory = gt_trajectory[:, :num_poses]  # [B, num_poses, 3]

                # ==================== 计算指标 ====================
                metrics = compute_metrics(traj_output, gt_trajectory)
                batch_ade = metrics['ade']
                batch_fde = metrics['fde']
                if pred_trajs is not None:
                    min_metrics = compute_minade_minfde_k(pred_trajs, gt_trajectory)
                    batch_minade_k = min_metrics["minade_k"]
                    batch_minfde_k = min_metrics["minfde_k"]
                else:
                    # 单模态情况下，min@K 退化为 top1
                    batch_minade_k = batch_ade
                    batch_minfde_k = batch_fde

                total_ade += batch_ade * B
                total_fde += batch_fde * B
                total_minade_k += batch_minade_k * B
                total_minfde_k += batch_minfde_k * B
                total_samples += B

            if batch_idx % 50 == 0:
                logger.info(f"Validation Epoch {epoch}, Batch {batch_idx}/{len(val_loader)}, "
                           f"ADE: {batch_ade:.4f}, FDE: {batch_fde:.4f}, "
                           f"minADE@K: {batch_minade_k:.4f}, minFDE@K: {batch_minfde_k:.4f}")

        except Exception as e:
            failed_batches += 1
            logger.warning(f"Validation batch {batch_idx} failed: {e}")
            continue

    if total_samples == 0:
        raise RuntimeError(
            f"Validation produced zero successful samples (failed_batches={failed_batches}). "
            "Please check model interfaces and validation data pipeline."
        )

    # 计算平均指标
    avg_ade = total_ade / total_samples if total_samples > 0 else 0.0
    avg_fde = total_fde / total_samples if total_samples > 0 else 0.0
    avg_minade_k = total_minade_k / total_samples if total_samples > 0 else 0.0
    avg_minfde_k = total_minfde_k / total_samples if total_samples > 0 else 0.0

    # 分布式同步: 聚合所有进程的指标
    if world_size > 1:
        # 转换为tensor便于同步（聚合总和，不聚合均值）
        metrics_tensor = torch.tensor([total_ade, total_fde, total_minade_k, total_minfde_k, total_samples],
                                       dtype=torch.float64, device=device)

        # All-reduce 聚合
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)

        # 重新计算平均值
        total_samples_all = metrics_tensor[4].item()
        if total_samples_all > 0:
            avg_ade = metrics_tensor[0].item() / total_samples_all
            avg_fde = metrics_tensor[1].item() / total_samples_all
            avg_minade_k = metrics_tensor[2].item() / total_samples_all
            avg_minfde_k = metrics_tensor[3].item() / total_samples_all

    # 恢复训练模式
    encoder_unwrapped.train()
    predictor_unwrapped.train()
    planner_unwrapped.train()

    return {
        'ade': avg_ade,
        'fde': avg_fde,
        'minade_k': avg_minade_k,
        'minfde_k': avg_minfde_k,
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
) -> dict:
    """
    运行验证的入口函数 (for train_giant.py)

    Args:
        encoder: 编码器 (可能是DDP包装)
        predictor: 预测器 (可能是DDP包装)
        planner: 规划器 (可能是DDP包装)
        val_loader: 验证数据加载器
        val_sampler: 验证数据采样器
        config: 配置字典
        epoch: 当前epoch
        rank: 当前进程rank
        world_size: world size

    Returns:
        metrics: 验证指标字典
    """
    # 设备配置（与 train_giant.py 保持一致）
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

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
    num_time_steps = num_poses

    # Loss 配置 (用于获取 normalize_reps)
    cfgs_loss = config.get("loss", {})
    normalize_reps = cfgs_loss.get("normalize_reps", True)
    cfgs_planner = config.get("planner", {})
    status_mode = cfgs_planner.get("status_mode", "last")
    z_ar_mode = cfgs_planner.get("z_ar_mode", "full")
    use_z_context = cfgs_planner.get("use_z_context", False)
    assert z_ar_mode in ("full", "first_step"), f"Invalid planner.z_ar_mode={z_ar_mode}"

    logger.info(f"Starting validation for epoch {epoch}...")
    logger.info(f"Validation config: tokens_per_frame={tokens_per_frame}, num_poses={num_poses}, "
                f"num_time_steps={num_time_steps}, normalize_reps={normalize_reps}, "
                f"status_mode={status_mode}, z_ar_mode={z_ar_mode}, use_z_context={use_z_context}")

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
        num_time_steps=num_time_steps,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        normalize_reps=normalize_reps,
        status_mode=status_mode,
        z_ar_mode=z_ar_mode,
        use_z_context=use_z_context,
    )

    if rank == 0:
        logger.info(f"="*50)
        logger.info(f"Validation Results - Epoch {epoch}:")
        logger.info(f"  ADE (Average Displacement Error): {metrics['ade']:.4f} m")
        logger.info(f"  FDE (Final Displacement Error):    {metrics['fde']:.4f} m")
        logger.info(f"  minADE@K:                         {metrics['minade_k']:.4f} m")
        logger.info(f"  minFDE@K:                         {metrics['minfde_k']:.4f} m")
        logger.info(f"="*50)

    return metrics
