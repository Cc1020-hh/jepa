# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import matplotlib.pyplot as plt
import math
import sys
sys.path.insert(0, '/disk/deepdata/hch_workspace/code/vjepa2/ddddetection_torchcv')
from app.vjepa_cowa.cowa import init_data_hf_temporal, init_data_seg,init_data_only_seg
from app.vjepa_droid.transforms import make_transforms
from app.vjepa_droid.utils import init_opt, init_opt_resample_world_model,init_opt_no_resample_world_model,init_video_model, load_checkpoint, load_pretrained, load_pretrained_safetensors,init_predictor_model
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from app.vjepa_cowa.RopeResample import RoPEPerceiverResampler
from transformers import AutoVideoProcessor, AutoModel
from torch.utils.tensorboard import SummaryWriter
from app.vjepa_cowa.seg_neck2 import SFP
from app.vjepa_cowa.seg_head2 import SimpleSemanticSegHead
# from ddddetection_torchcv.torchcv.modeling.head.semantic_seg_head import SimpleSemanticSegHead
from  app.vjepa_cowa.co_detr_decoder import CoDetrDecoder
from Drive_JEPA.navsim_v1.navsim.agents.drive_jepa_perception_free.drive_jepa_model import TrajectoryHead
# ==================== 新增：评估模块导入 ====================
from app.vjepa_cowa_planner.val import run_validation
# ==================== 新增：Seg Head相关导入 ====================
# from ddddet.modeling.utils.weight_init import c2_xavier_fill

log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


# ==================== 方案三：单帧预测 Planner ====================
class SingleFramePlanner(nn.Module):
    """
    单帧预测范式：从当前时刻的表征预测未来轨迹
    参考 Drive-JEPA 的设计

    关键改进：
    1. 只使用最后时刻的表征
    2. 添加位置编码 (keyval_embedding)
    3. 结合状态信息 (status_encoding)
    """

    def __init__(
        self,
        encoder_dim: int = 1024,
        tf_d_model: int = 256,
        tf_d_ffn: int = 1024,
        tf_num_layers: int = 3,
        tf_num_head: int = 8,
        tf_dropout: float = 0.0,
        tokens_per_frame: int = 256,
        num_poses: int = 7,
        status_dim: int = 13,
        use_spatial_tokens: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.tf_d_model = tf_d_model
        self.tokens_per_frame = tokens_per_frame
        self.num_poses = num_poses
        self.use_spatial_tokens = use_spatial_tokens

        # 1. 图像特征投影
        if use_spatial_tokens:
            # 保留空间 token
            self.image_fc = nn.Linear(encoder_dim, tf_d_model)
            num_keyval = tokens_per_frame + 1  # 空间token + 状态
        else:
            # 池化后只有一个向量
            self.image_fc = nn.Linear(encoder_dim, tf_d_model)
            num_keyval = 2  # 图像 + 状态

        # 2. 状态编码
        self.status_encoding = nn.Sequential(
            nn.Linear(status_dim, 128),
            nn.ReLU(),
            nn.Linear(128, tf_d_model),
        )

        # 3. Key-Value 位置编码 (关键！)
        self.keyval_embedding = nn.Embedding(num_keyval, tf_d_model)

        # 4. Query 位置编码 (轨迹点)
        self.query_embedding = nn.Embedding(num_poses, tf_d_model)

        # 5. Transformer Decoder
        self.transformer = nn.Transformer(
            d_model=tf_d_model,
            nhead=tf_num_head,
            num_encoder_layers=tf_num_layers,
            num_decoder_layers=tf_num_layers,
            dim_feedforward=tf_d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )

        # 6. 轨迹预测头
        self.trajectory_head = TrajectoryHead(num_poses, tf_d_ffn, tf_d_model)

    def forward(self, z_ar: torch.Tensor, status_feature: torch.Tensor):
        """
        Args:
            z_ar: Predictor 自回归输出 [B, T*tokens_per_frame, encoder_dim]
            status_feature: 当前状态 [B, status_dim]
        Returns:
            {"trajectory": [B, num_poses, 3]}  (x, y, yaw)
        """
        B = z_ar.shape[0]

        # Step 1: 提取最后时刻的表征
        z_last = z_ar[:, :self.tokens_per_frame]  # [B, tokens_per_frame, encoder_dim]

        if self.use_spatial_tokens:
            # 方案B: 保留所有空间 token
            img_feat = self.image_fc(z_last)  # [B, tokens_per_frame, tf_d_model]
        else:
            # 方案A: 空间池化
            z_pooled = z_last.mean(dim=1)  # [B, encoder_dim]
            img_feat = self.image_fc(z_pooled)  # [B, tf_d_model]
            img_feat = img_feat.unsqueeze(1)  # [B, 1, tf_d_model]

        # Step 2: 编码状态信息
        status_encoded = self.status_encoding(status_feature)  # [B, tf_d_model]
        status_encoded = status_encoded.unsqueeze(1)  # [B, 1, tf_d_model]

        # Step 3: 组合 keyval
        keyval = torch.cat([img_feat, status_encoded], dim=1)  # [B, num_keyval, tf_d_model]

        # Step 4: 添加位置编码 (关键！)
        num_keyval = keyval.shape[1]
        keyval = keyval + self.keyval_embedding.weight[:num_keyval, :].unsqueeze(0)

        # Step 5: Transformer 解码
        query = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, num_poses, tf_d_model]
        query_out = self.transformer(src=keyval, tgt=query)  # [B, num_poses, tf_d_model]

        # Step 6: 轨迹预测
        trajectory = self.trajectory_head(query_out)  # {"trajectory": [B, num_poses, 3]}

        return trajectory


def l1_length_normalized_loss(pred, gt, alpha=5.0, eps=1e-6):
    """
    长度归一化的 L1 损失 (参考 Drive-JEPA)
    避免短轨迹和长轨迹的权重不平衡
    """
    per_sample_l1 = (pred - gt).abs().mean(dim=[1, 2])  # [B]
    dxy = gt[:, 1:, :2] - gt[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = w * (w.numel() / (w.sum() + eps))
    return (w * per_sample_l1).mean()


def prepare_status_feature(states, actions):
    """从 states 和 actions 提取状态特征"""
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


def prepare_status_feature_with_history(states, actions, history_len=5):
    """
    改进版：包含历史轨迹信息

    输出: [B, D] (D 可配置)
    """
    B = states.shape[0]
    device = states.device

    # ========== 1. 当前状态 ==========
    current_vel = states[:, -1, 6:7]      # [B, 1]
    current_yaw = states[:, -1, 5:6]      # [B, 1]
    current_xy = states[:, -1, 0:2]       # [B, 2]

    # ========== 2. 运动特征 ==========
    # 加速度
    acc = states[:, -1, 6:7] - states[:, -2, 6:7]  # [B, 1]

    # 角速度 (转向率) - 关键特征!
    yaw_rate = states[:, -1, 5:6] - states[:, -2, 5:6]  # [B, 1]

    # ========== 3. 历史轨迹特征 (新增!) ==========
    history_xy = states[:, -history_len:, 0:2]  # [B, history_len, 2]

    # 3.1 历史轨迹位移 (整体移动方向)
    displacement = history_xy[:, -1] - history_xy[:, 0]  # [B, 2]

    # 3.2 轨迹曲率 (转弯程度) - 关键!
    yaw_change = states[:, -1, 5] - states[:, -history_len, 5]
    travel_dist = torch.norm(displacement, dim=-1) + 1e-6
    curvature = (yaw_change / travel_dist).unsqueeze(-1)  # [B, 1]

    # 3.3 横向摆动 (变道意图检测)
    lateral_shift = torch.std(history_xy[:, :, 1], dim=-1, keepdim=True)  # [B, 1]

    # 3.4 速度变化趋势
    history_vel = states[:, -history_len:, 6]  # [B, history_len]
    vel_trend = (history_vel[:, -1] - history_vel[:, 0]).unsqueeze(-1)  # [B, 1]

    # ========== 4. 动作特征 ==========
    last_action = actions[:, -1, :2] if actions.shape[1] > 0 else torch.zeros(B, 2, device=device)

    # ========== 5. 拼接 ==========
    status_feature = torch.cat([
        current_vel,      # 速度
        current_yaw,      # 朝向
        current_xy,       # 位置
        acc,              # 加速度
        yaw_rate,         # 角速度 (关键!)
        displacement,     # 历史位移
        curvature,        # 曲率 (关键!)
        lateral_shift,    # 横向摆动
        vel_trend,        # 速度趋势
        last_action,      # 最后动作
    ], dim=-1)  # [B, 13]

    return status_feature


def prepare_seg_features(
    context_clips, 
    seg_targets, 
    z_perceiver,
    seg_neck, 
    tubelet_size, 
    tokens_per_frame, 
    device, 
    mixed_precision, 
    dtype,
    normalize_reps=False
):
    """
    统一的前向特征提取函数。
    负责：Encoder -> Perceiver -> Reshape -> 提取Query/Memory -> Neck -> 输出Head的输入特征
    """

    # 记录用于可视化的元数据 (保留原始图片引用和对应索引)
    vis_meta = [] 

    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        # 1. Encoder 前向

        # 3. Reshape 逻辑 (统一在此处管理)
        B, Total_Latents, D = z_perceiver.shape
        num_frames_latent = Total_Latents // tokens_per_frame
        # logger.info(f"preparing seg_feature num_frames_latent: {num_frames_latent}")
        # Encoder Output Reshape
        # breakpoint()
        enc_tokens = z_perceiver.shape[1]
        enc_dim = z_perceiver.shape[-1]
        spatial_tokens = enc_tokens // num_frames_latent
        r_size_enc = int(math.sqrt(spatial_tokens)) 

        # [B, T_latent, H_feat, W_feat, D]
        z_perceiver_reshaped = z_perceiver.view(B, num_frames_latent, r_size_enc, r_size_enc, enc_dim)

        # 4. 收集 Batch 数据
        batch_queries_list = []    
        batch_targets_list = []    

        for b in range(B):
            if seg_targets[b] is None: continue
            masks_k, indices_k = seg_targets[b]

            for k in range(len(indices_k)):
                t_frame = indices_k[k].item()
                t_latent = t_frame // tubelet_size

                if t_latent >= num_frames_latent: continue

                # A. 提取 Perceiver Latent -> 作为 Query
                # [Tokens_per_frame, D] -> [D, Tokens_per_frame] -> (Reshape logic if needed)
                # 这里保持你原有的逻辑:
                query_slice = z_perceiver_reshaped[b, t_latent].permute(2, 0, 1) 

                # B. 收集
                batch_queries_list.append(query_slice)

                # C. Target 准备 (需要在外部定义好 prepare_targets 或者简单封装)
                target_k = {
                    'labels': torch.ones(masks_k[k].shape[0], dtype=torch.long, device=device),
                    'masks': masks_k[k].float().to(device)
                }
                batch_targets_list.append(target_k)

                # D. 收集可视化元数据
                vis_meta.append({
                    'batch_idx': b,
                    't_frame': t_frame,
                    'gt_mask': masks_k[k].cpu(), # 此时先转CPU省显存
                    'img_tensor': context_clips[b, :, t_frame, :, :].cpu() # 原图
                })

        # 5. 堆叠与过 Neck
        valid_samples = len(batch_queries_list)
        neck_out = None
        batched_targets = {}

        if valid_samples > 0:
            batched_queries = torch.stack(batch_queries_list) # [N, D, H, W]

            # 堆叠 Targets
            example_target = batch_targets_list[0]
            for key in example_target.keys():
                values = [d[key] for d in batch_targets_list]
                if isinstance(values[0], torch.Tensor):
                    stacked_val = torch.stack(values)
                    if stacked_val.dim() == 4 and stacked_val.shape[1] == 1:
                        stacked_val = stacked_val.squeeze(1)
                    batched_targets[key] = stacked_val
                else:
                    batched_targets[key] = values

            # 进入 Neck
            neck_out = seg_neck(batched_queries)


    return neck_out, batched_targets, valid_samples, vis_meta

def save_training_visualization(
    pred_results, 
    vis_meta, 
    output_dir, 
    epoch, 
    itr, 
    limit=5
):
    """
    统一的可视化绘图函数
    """
    os.makedirs(output_dir, exist_ok=True)
    # 取最后一层输出，通常是 [N, num_classes, H, W]
    pred_raw_batch = pred_results[-1] 

    count = 0
    for i, meta in enumerate(vis_meta):
        if count >= limit: break

        # 1. 处理预测 Mask
        pred_raw = pred_raw_batch[i] # [num_classes, H, W]

        # 简单处理：Sigmoid + Argmax (假设你的逻辑是这样)
        mask_cat = torch.argmax(pred_raw.sigmoid(), dim=0) # 注意 dim可能是0或1，取决于你的shape
        # 这里假设 pred_raw 是 [C, H, W]，则 dim=0

        mask_cat = mask_cat.cpu().numpy().astype(np.uint8)
        mask_cat[mask_cat == 0] = 0 # 背景
        mask_cat[mask_cat == 1] = 255 # 前景 (假设二分类)

        # 2. 处理 GT
        gt_tensor = meta['gt_mask']
        if gt_tensor.dim() == 3:
            gt_merged_mask, _ = torch.max(gt_tensor, dim=0)
        else:
            gt_merged_mask = gt_tensor
        gt_vis = gt_merged_mask.numpy()

        # 3. 处理原图
        img_tensor = meta['img_tensor']
        img_vis = img_tensor.permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)

        # 4. 绘图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_vis)
        axes[0].set_title(f"E{epoch}_I{itr} Frame {meta['t_frame']}")
        axes[0].axis('off')

        axes[1].imshow(gt_vis, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        axes[2].imshow(mask_cat, cmap='gray')
        axes[2].set_title("Prediction")
        axes[2].axis('off')

        save_path = os.path.join(output_dir, f"vis_E{epoch}_I{itr}_s{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        count += 1

def visualize_trajectory(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    output_dir: str,
    epoch: int,
    itr: int,
    limit: int = 5,
):
    """
    可视化预测轨迹和真实轨迹

    Args:
        pred_traj: 预测轨迹 [B, num_poses, 3] (x, y, yaw)
        gt_traj: 真实轨迹 [B, num_poses, 3] (x, y, yaw)
        output_dir: 输出目录
        epoch: 当前 epoch
        itr: 当前 iteration
        limit: 最多可视化多少个样本
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_traj = pred_traj.detach().cpu().float().numpy()
    gt_traj = gt_traj.detach().cpu().float().numpy()

    batch_size = min(pred_traj.shape[0], limit)
    num_poses = pred_traj.shape[1]

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制起点 (0, 0) - 车辆当前位置
        ax.scatter(0, 0, c='red', s=100, marker='*', label='Ego (Start)', zorder=5)

        # 绘制真实轨迹
        gt_x = gt_traj[i, :, 0]
        gt_y = gt_traj[i, :, 1]
        gt_yaw = gt_traj[i, :, 2]

        # 绘制轨迹线
        ax.plot(gt_x, gt_y, 'g-', linewidth=2, label=f'Ground Truth ({num_poses} pts)', alpha=0.8)
        ax.scatter(gt_x, gt_y, c='green', s=30, alpha=0.6)

        # 绘制真实轨迹的方向箭头
        arrow_step = max(1, len(gt_x) // 4)  # 最多绘制4个箭头
        for j in range(0, len(gt_x), arrow_step):
            dx = 0.3 * np.cos(gt_yaw[j])
            dy = 0.3 * np.sin(gt_yaw[j])
            ax.arrow(gt_x[j], gt_y[j], dx, dy,
                    head_width=0.15, head_length=0.1, fc='green', ec='green', alpha=0.7)

        # 绘制预测轨迹
        pred_x = pred_traj[i, :, 0]
        pred_y = pred_traj[i, :, 1]
        pred_yaw = pred_traj[i, :, 2]

        # 检查预测值是否有异常
        has_nan = np.any(np.isnan(pred_x)) or np.any(np.isnan(pred_y))
        has_inf = np.any(np.isinf(pred_x)) or np.any(np.isinf(pred_y))
        max_abs = max(np.abs(pred_x).max(), np.abs(pred_y).max())

        ax.plot(pred_x, pred_y, 'b-', linewidth=2, label=f'Prediction ({num_poses} pts)', alpha=0.8)
        ax.scatter(pred_x, pred_y, c='blue', s=30, alpha=0.6)

        # 绘制预测轨迹的方向箭头
        for j in range(0, len(pred_x), arrow_step):
            dx = 0.3 * np.cos(pred_yaw[j])
            dy = 0.3 * np.sin(pred_yaw[j])
            ax.arrow(pred_x[j], pred_y[j], dx, dy,
                    head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.7)

        # 计算 L2 误差
        l2_error = np.sqrt(((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).mean())
        yaw_error = np.abs(pred_yaw - gt_yaw).mean()

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # 标题包含调试信息
        debug_info = f'NaN: {has_nan}, Inf: {has_inf}, MaxAbs: {max_abs:.2f}'
        ax.set_title(f'Trajectory Visualization\nEpoch {epoch}, Iter {itr}, Sample {i}\n'
                    f'L2 Error: {l2_error:.3f}m, Yaw Error: {yaw_error:.3f}rad\n'
                    f'[{debug_info}]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # 动态调整坐标轴范围（基于GT轨迹，避免异常预测值影响）
        all_x = np.concatenate([gt_x, [0]])
        all_y = np.concatenate([gt_y, [0]])

        # 如果预测值正常，也纳入范围计算
        if max_abs < 100:  # 阈值：100米
            all_x = np.concatenate([all_x, pred_x])
            all_y = np.concatenate([all_y, pred_y])

        margin = max(1.0, (all_x.max() - all_x.min()) * 0.1)
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        save_path = os.path.join(output_dir, f"traj_E{epoch}_I{itr}_s{i}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    p_repo = cfgs_meta.get("pretrain_repo", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    use_extrinsics = cfgs_model.get("use_extrinsics", False)
    # --train
    cfgs_train = args.get("train")
    encoder_train = cfgs_train.get("encoder_train",False)
    student_perceiver_train = cfgs_train.get("student_perceiver",True)
    seg_head_train = cfgs_train.get("seg_head",True)
    encoder_ema = cfgs_train.get("encoder_ema",False)
    perceiver_ema = cfgs_train.get("perceiver_ema",True)
    predictor_train = cfgs_train.get("predictor_train",True)
    # -- EMA
    cfgs_ema = args.get("ema")
    use_ema = cfgs_ema.get("use_ema", True)
    perceiver_num_latents = cfgs_ema.get("perceiver_num_latents", 128)
    perceiver_depth = cfgs_ema.get("perceiver_depth", 5)
    perceiver_num_heads = cfgs_ema.get("perceiver_num_heads", 16)
    perceiver_head_dim = cfgs_ema.get("perceiver_head_dim", 64)
    ema_start = cfgs_ema.get("ema_start", 0.996)  # 默认起始值
    ema_end = cfgs_ema.get("ema_end", 0.999)      # 默认结束值
    ema = [ema_start, ema_end]  # 与 train_giant.py 保持一致的格式
    resample_size = cfgs_ema.get("resample_size", 16)

    # ==================== 新增：Segmentation配置 ====================
    cfgs_seg = args.get("segmentation", {})
    use_segmentation = cfgs_seg.get("use_segmentation", True)
    # 移除 num_seg_classes，因为不再需要分类
    seg_loss_weight = cfgs_seg.get("seg_loss_weight", 1.0)
    seg_loss_weights = cfgs_seg.get("loss_weights", {
        "loss_mask": 5.0,
        "loss_dice": 5.0,
    })
    # -- PLANNER (方案三：单帧预测)
    cfgs_planner = args.get("planner", {})
    tf_d_model = cfgs_planner.get("tf_d_model", 256)
    tf_d_ffn = cfgs_planner.get("tf_d_ffn", 1024)
    tf_num_layers = cfgs_planner.get("tf_num_layers", 3)
    tf_num_head = cfgs_planner.get("tf_num_head", 8)
    tf_dropout = cfgs_planner.get("tf_dropout", 0.0)
    planner_loss_weight = cfgs_planner.get("planner_loss_weight", 1.0)
    use_spatial_tokens = cfgs_planner.get("use_spatial_tokens", False)  # 新增：是否使用空间token

    # -- DATA
    cfgs_data = args.get("data")
    datasets = cfgs_data.get("datasets", [])
    dataset_path = datasets[0]
    # -- 验证数据集配置
    val_datasets = cfgs_data.get("val_datasets", None)  # 验证数据集路径列表
    val_dataset_path = val_datasets[0] if val_datasets else None
    val_freq = cfgs_meta.get("val_freq", 5)  # 验证频率 (每隔多少个epoch验证一次)
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    camera_frame = cfgs_data.get("camera_frame", False)
    camera_views = cfgs_data.get("camera_views", ["left_mp4_path"])
    stereo_view = cfgs_data.get("stereo_view", False)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    target_frame = cfgs_data.get("num_target_frames", 16)
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    horizontal_flip = cfgs_data_aug.get("horizontal_flip", False)
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp")
    normalize_reps = cfgs_loss.get("normalize_reps")
    auto_steps = min(cfgs_loss.get("auto_steps", 1), max_num_frames)
    tokens_per_frame = int(int((crop_size // patch_size) ** 2))

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # ----------------------------------------------------------------------- #
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
                f"torch.cuda.current_device()={torch.cuda.current_device()}, "
                f"torch.cuda.device_count()={torch.cuda.device_count()}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "seg_loss"),
        ("%.5f", "mask_loss"),  # 新增
        ("%.5f", "dice_loss"),  # 新增
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )
    
    if rank == 0:
        tensorboard_dir = os.path.join(folder, "tensorboard_logs")
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    else:
        tb_writer = None

    # -- 初始化模型
    logger.info(f"begin init automodel:")
    local_path = p_repo
    model = AutoModel.from_pretrained(local_path, local_files_only=True)
    processor = AutoVideoProcessor.from_pretrained(local_path, local_files_only=True)
    logger.info(f"end init automodel: ")
    logger.info(f"begin init predictor:")
    predictor = init_predictor_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=512,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        action_embed_dim=7,
        pred_is_frame_causal=pred_is_frame_causal,
        use_extrinsics=use_extrinsics,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
        use_perceiver_ema=perceiver_ema,
        # resample_size=resample_size
        target_shape = None
    )
    
    encoder = model.encoder
    target_encoder = copy.deepcopy(encoder)
    
    # 打印参数量
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"init encoder_params: {encoder_params / 1e6:>8.2f}M")
    target_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    print(f"init target_encoder_params: {target_encoder_params / 1e6:>8.2f}M")
    predictor_params = sum(p.numel() for p in predictor.parameters())
    print(f"init predictor_params: {predictor_params / 1e6:>8.2f}M")
    
    
    # 移动模型到GPU
    encoder = encoder.to(device)
    target_encoder = target_encoder.to(device)
    model = model.to(device)

    if use_segmentation:
        seg_neck = SFP(
            input_channels = [1024],
            out_channels = 256,
            use_p2 = True,
            use_act_checkpoint = False
        )
        seg_head = SimpleSemanticSegHead(
            input_strides = [4,8,16,32,64],
            num_classes= 2 ,#占位
            decoder= CoDetrDecoder(
                num_proposals = 1500,
                embed_dims = 256,
                num_heads = 8,
                num_levels = 5,
                dropout = 0.0,
                feedforward_channels = 2048,
                ffn_dropout = 0.0,
                num_layers = 6,
                return_intermediate = True,
                two_stage = False,
                num_co_heads = 0,
                with_coord_feat = False,
                with_pos_coord = False
            ),
            embed_dims = 256,
            out_mask_dim= 256, #占位
            loss_weights={
            "loss_seg": 2.0,
            "loss_dice": 5.0,
        },
            subcat_num=0,

        )
        seg_neck = seg_neck.to(device)
        seg_head = seg_head.to(device)
    else:
        seg_head = None
        seg_neck = None
    # -- PLANNER (方案三：单帧预测)
    num_poses = (target_frame // tubelet_size) - 1  # 时间压缩后的轨迹点数
    encoder_dim = encoder.config.hidden_size  # ViT的embed dim，比如1024

    planner = SingleFramePlanner(
        encoder_dim=encoder_dim,
        tf_d_model=tf_d_model,
        tf_d_ffn=tf_d_ffn,
        tf_num_layers=tf_num_layers,
        tf_num_head=tf_num_head,
        tf_dropout=tf_dropout,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        status_dim=8,
        use_spatial_tokens=use_spatial_tokens,
    ).to(device)

    planner_params = sum(p.numel() for p in planner.parameters())
    logger.info(f"planner_params: {planner_params / 1e6:.2f}M (use_spatial_tokens={use_spatial_tokens})")

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        target_encoder.compile()
        predictor.compile()
        if seg_head is not None:
            seg_head.compile()

    # -- 数据加载
    video_collator = torch.utils.data.default_collate
    transform = make_transforms(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    seg_data_root = cfgs_seg.get("seg_data_root", "/disk/deepdata/dataset/nvs/data/sam3_autolabeling_point")

    # 初始化数据加载器
    (unsupervised_loader, unsupervised_sampler) = init_data_only_seg(
        data_path=dataset_path,
        batch_size=batch_size,
        fps=fps,
        camera_views=camera_views,
        camera_frame=camera_frame,
        frames_per_clip= target_frame,
        stereo_view=stereo_view,
        tubelet_size = tubelet_size,
        transform=transform,
        collator=None,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        load_segmentation=use_segmentation,  # 新增
        seg_data_root=seg_data_root,  # 新增
        crop_size= crop_size
    )
    
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # ==================== 初始化验证数据加载器 ====================
    val_loader = None
    val_sampler = None
    if val_dataset_path is not None and os.path.exists(val_dataset_path):
        logger.info(f"Initializing validation dataset from: {val_dataset_path}")
        (val_loader, val_sampler) = init_data_only_seg(
            data_path=val_dataset_path,
            batch_size=batch_size,
            fps=fps,
            camera_views=camera_views,
            camera_frame=camera_frame,
            frames_per_clip=target_frame,
            stereo_view=stereo_view,
            tubelet_size=tubelet_size,
            transform=transform,  # 使用相同的transform
            collator=None,
            num_workers=num_workers,
            world_size=world_size,
            pin_mem=pin_mem,
            persistent_workers=persistent_workers,
            rank=rank,
            load_segmentation=False,  # 验证时不需要分割标注
            seg_data_root=None,
            crop_size=crop_size
        )
        logger.info(f"Validation dataset initialized with {len(val_loader)} batches")
    else:
        logger.warning(f"Validation dataset not configured or path does not exist. Validation will be skipped.")
        if val_dataset_path:
            logger.warning(f"  Provided path: {val_dataset_path}")

    # -- 优化器（添加seg_head参数）
    optimizer, scaler, scheduler, wd_scheduler = init_opt_no_resample_world_model(
        encoder= encoder,
        predictor=predictor,
        # student_perceiver=student_perceiver,
        seg_head=seg_head if use_segmentation else None,  # 新增
        seg_neck=seg_neck,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    # 添加 planner 参数
    optimizer.add_param_group({
        "params": planner.parameters(),
        "lr": lr,
        "weight_decay": wd,
    })
    # DDP包装
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=False, find_unused_parameters=True)
    # student_perceiver = DistributedDataParallel(student_perceiver, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)

    if seg_head is not None:
        seg_head = DistributedDataParallel(seg_head, find_unused_parameters=False)
    if seg_neck is not None:
        seg_neck = DistributedDataParallel(seg_neck, find_unused_parameters=False)

    planner = DistributedDataParallel(planner, find_unused_parameters=False)

    # 冻结参数
    for p in encoder.parameters():
        if encoder_train:
            p.requires_grad = True
        else:
            p.requires_grad = False
    for p in target_encoder.parameters():
        p.requires_grad = False
    # for p in teacher_perceiver.parameters():
    #     p.requires_grad = False
    # for p in student_perceiver.parameters():
    #     if student_perceiver_train:
    #         p.requires_grad = True
    #     else:
    #         p.requires_grad = False
    for p in predictor.parameters():
        if predictor_train:
            p.requires_grad = True
        else:
            p.requires_grad = False
    if seg_head is not None:
        for p in seg_head.parameters():
            if seg_head_train:
                p.requires_grad = True
            else:
                p.requires_grad = False
    if seg_head is not None:
        for p in seg_neck.parameters():
            if seg_head_train:
                p.requires_grad = True
            else:
                p.requires_grad = False
    # 统计所有参与梯度优化的参数量
    total_trainable_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.requires_grad:
                total_trainable_params += p.numel()

    # 分模块统计
    planner_params_total = sum(p.numel() for p in planner.parameters() if p.requires_grad)

    logger.info(f"{'='*50}")
    logger.info(f"Trainable Parameters Summary:")
    logger.info(f"  encoder:           {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    logger.info(f"  predictor:         {sum(p.numel() for p in predictor.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    if seg_neck is not None:
        logger.info(f"  seg_neck:          {sum(p.numel() for p in seg_neck.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    if seg_head is not None:
        logger.info(f"  seg_head:          {sum(p.numel() for p in seg_head.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    logger.info(f"  planner:           {planner_params_total / 1e6:>8.2f}M")
    logger.info(f"{'─'*50}")
    logger.info(f"  ALL trainable:     {total_trainable_params / 1e6:>8.2f}M")
    logger.info(f"{'='*50}")

    # 加载预训练权重
    _, predictor, _ = load_pretrained_safetensors(
        r_path=p_file,
        predictor=predictor,
        load_predictor=load_predictor,
        load_encoder=False
    )
    # @torch.no_grad()
    # def update_teacher_perceiver():
    #     """使用EMA更新teacher perceiver的参数"""
    #     student_state = student_perceiver.module.state_dict() if hasattr(student_perceiver, 'module') else student_perceiver.state_dict()
    #     teacher_state = teacher_perceiver.state_dict()

    #     for key in student_state:
    #         teacher_state[key] = ema_decay * teacher_state[key] + (1 - ema_decay) * student_state[key]

    #     teacher_perceiver.load_state_dict(teacher_state)

    # -- momentum schedule (动态 EMA，与 train_giant.py 保持一致)
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs)
        for i in range(int(ipe * num_epochs) + 1)
    )

    @torch.no_grad()
    def update_teacher_encoder(m):
        """
        使用动态 momentum 更新 teacher encoder 的参数
        采用高效的原地操作方式 (与 train_giant.py 一致)

        :param m: 当前的 momentum 值 (从 momentum_scheduler 获取)
        """
        # 收集所有参数
        params_k = []
        params_q = []
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            params_k.append(param_k)
            params_q.append(param_q)
        # 高效的原地操作
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1 - m)
    start_epoch = 0

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            # "student_perceiver": student_perceiver.state_dict(),
            # "teacher_perceiver": teacher_perceiver.state_dict(),
            "opt": optimizer.state_dict(),
                        "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        
        # ==================== 新增：保存seg_head ====================
        if seg_head is not None:
            save_dict["seg_head"] = seg_head.state_dict()
        if seg_neck is not None:
            save_dict["seg_neck"] = seg_neck.state_dict()
        save_dict["planner"] = planner.state_dict()
        try:
            torch.save(save_dict, path)
            logger.info(f"主人，checkpoint已保存到: {path}")
        except Exception as e:
            logger.info(f"主人，保存checkpoint时遇到异常: {e}")

    logger.info("主人，正在初始化数据加载器...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f"主人，跳过 {skip_batches} 个批次")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"跳过 {itr}/{skip_batches} 批次")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    # ==================== 训练循环 ====================
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"主人，开始训练 Epoch {epoch + 1}")

        loss_meter = AverageMeter()
        jloss_meter = AverageMeter()
        sloss_meter = AverageMeter()
        seg_loss_meter = AverageMeter()  # 新增
        mask_loss_meter = AverageMeter()  # 新增
        dice_loss_meter = AverageMeter()  # 新增
        traj_loss_meter = AverageMeter() # planner
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()
            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("主人，数据加载器已耗尽，正在刷新...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"主人，加载数据时遇到异常 (重试次数 {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"主人，超过最大重试次数 ({NUM_RETRIES})，跳过此批次。")
                        raise e

            # ==================== 数据加载 ====================
            def load_clips():
                context_frames = sample[0].to(device, non_blocking=True)  # [B, C, 2, H, W]
                actions = sample[1].to(device, dtype=torch.float, non_blocking=True)  # [B, 15, 7]
                states = sample[2].to(device, dtype=torch.float, non_blocking=True)   # [B, 16, 7]
                extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)  # [B, 16, 7]
                
                # ==================== 新增：加载分割标注 ====================
                seg_targets = None
                if use_segmentation and len(sample) > 4:
                    # 假设sample[4]是分割标注
                    # 格式: list of dict, 每个dict包含 'labels' 和 'masks'
                    seg_targets = sample[4]
                    # 如果seg_targets不在device上，移动到device
                    if isinstance(seg_targets, list):
                        for i in range(len(seg_targets)):
                            if 'labels' in seg_targets[i]:
                                seg_targets[i]['labels'] = seg_targets[i]['labels'].to(device, non_blocking=True)
                            if 'masks' in seg_targets[i]:
                                seg_targets[i]['masks'] = seg_targets[i]['masks'].to(device, non_blocking=True)
                    # seg is Int8
                return context_frames, actions, states, extrinsics, seg_targets

            context_clips, actions, states, extrinsics, seg_targets = load_clips()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("主人，正在运行垃圾回收...")
                gc.collect()
            VISUALIZE_FREQ = 100 # 根据需要调整
            should_visualize = (rank == 0) and (itr % VISUALIZE_FREQ == 0)
            vis_output_dir = os.path.join(folder, "train_vis_debug")
            
            # ==================== 训练步骤 ====================
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(target_clips):
                    """Teacher分支：encoder -> perceiver -> 输出"""
                    with torch.no_grad():
                        B, C, T, H, W = target_clips.shape
                        assert C == 3, f"Expected 3 channels (RGB), got {C}"
                        assert T % tubelet_size == 0, f"T={T} not divisible by tubelet_size={tubelet_size}"

                        # Encoder (frozen)
                        h_target = target_encoder(target_clips.permute(0, 2, 1, 3, 4))
                        h_ = h_target.last_hidden_state

                        # Teacher Perceiver (frozen, no grad)
                        # h = teacher_perceiver(h_)  # [B, num_latents, embed_dim]

                        if normalize_reps:
                            h_ = F.layer_norm(h_, (h_.size(-1),))

                        return h_

                def forward_context(context_clips):
                    """Student分支：encoder -> perceiver -> 输出"""
                    B, C, T, H, W = context_clips.shape
                    assert C == 3

                    with torch.no_grad():
                        B, C, T, H, W = context_clips.shape
                        assert C == 3, f"Expected 3 channels (RGB), got {C}"
                        assert T % tubelet_size == 0, f"T={T} not divisible by tubelet_size={tubelet_size}"
                        # Encoder (frozen)
                        z_context = encoder(context_clips.permute(0, 2, 1, 3, 4))
                        z = z_context.last_hidden_state  # [B, N, D]

                    # Student Perceiver (trainable)
                    # z = student_perceiver(z)  # [B, num_latents, embed_dim]

                    if normalize_reps:
                        z = F.layer_norm(z, (z.size(-1),))

                    return z

                def forward_predictions(z, actions, states, extrinsics):
                    """Predictor保持不变"""
                    def _step_predictor(_z, _a, _s, _e):
                        _z = predictor(_z, _a, _s, _e)
                        if normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z
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
                    return z_tf, z_ar

                def loss_fn(z, h):
                    _h = h[:, tokens_per_frame : z.size(1) + tokens_per_frame]
                    return torch.mean(torch.abs(z - _h) ** loss_exp) / loss_exp

                # ==================== Forward pass ====================
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # 1. Teacher分支
                    h_target = forward_target(context_clips)
                    
                    # 2. Student分支
                    z_context = forward_context(context_clips)
                    z_pred = z_context
                    # 3. Predictor分支
                    z_tf, z_ar = forward_predictions(z_pred, actions, states, extrinsics)
                    
                    # 4. 计算JEPA损失
                    sloss = loss_fn(z_ar, h_target)
                    jloss = loss_fn(z_tf, h_target)
                    jepa_loss = jloss + sloss

                    # ==================== Planner (方案三：单帧预测) ====================
                    B = z_ar.shape[0]

                    # 准备状态特征 (使用历史轨迹信息)
                    status_feature = prepare_status_feature_with_history(states, actions)

                    # Planner forward
                    traj_output_dict = planner(z_ar, status_feature)
                    traj_output = traj_output_dict["trajectory"]  # [B, num_poses, 3]

                    # GT 轨迹转换（世界坐标 -> ego相对坐标）
                    StateSE2_indices = [0, 1, 5]
                    states_se2 = states[:, :, StateSE2_indices]  # [B, T, 3], 世界坐标

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

                    # 验证（只在前几步打印）
                    if itr < 20:
                        logger.info(
                            f"GT ego: "
                            f"x=[{ego_x.min():.2f}, {ego_x.max():.2f}], "
                            f"y=[{ego_y.min():.2f}, {ego_y.max():.2f}], "
                            f"yaw=[{ego_yaw.min():.3f}, {ego_yaw.max():.3f}], "
                            f"gt_trajectory.mean={gt_trajectory.abs().mean():.4f}"
                        )

                    # 使用长度归一化 L1 Loss
                    traj_loss = l1_length_normalized_loss(traj_output, gt_trajectory, alpha=5.0)

                    # 轨迹可视化
                    if should_visualize:
                        visualize_trajectory(
                            pred_traj=traj_output,
                            gt_traj=gt_trajectory,
                            output_dir=vis_output_dir,
                            epoch=epoch,
                            itr=itr,
                            limit=5
                        )

                    # 总loss
                    loss = jepa_loss + (planner_loss_weight * traj_loss)
                    # ==================== 新增：分割损失 ====================
                    seg_loss_value = 0.0
                    mask_loss_value = 0.0
                    dice_loss_value = 0.0
                    if use_segmentation and seg_head is not None:
                        # 检查是否有有效标注
                        neck_out, batched_targets, valid_samples, vis_meta = prepare_seg_features(
                        context_clips=context_clips,
                        seg_targets=seg_targets,
                        z_perceiver = z_context,
                        seg_neck=seg_neck,
                        tubelet_size=tubelet_size,
                        tokens_per_frame=tokens_per_frame,
                        device=device,
                        mixed_precision=mixed_precision,
                        dtype=dtype,
                        normalize_reps=normalize_reps
                    )

                    if use_segmentation and seg_head is not None and valid_samples > 0:
                        # --- 2. 计算 Training Loss ---
                        loss_dict = seg_head.module.get_loss(
                            inputs=neck_out,
                            targets=batched_targets,
                            input_query=None
                        )
                        total_seg_loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
                        seg_loss = total_seg_loss / valid_samples
                        loss = loss + (seg_loss_weight * seg_loss)
                        # 记录 Loss (保持原有逻辑)
                        with torch.no_grad():
                            last_layer_idx = 6 
                            seg_loss_value = loss_dict.get(f'loss_seg_{last_layer_idx}', torch.tensor(0.0)).detach().item()
                            dice_loss_value = loss_dict.get(f'loss_dice_{last_layer_idx}', torch.tensor(0.0)).detach().item()
                            mask_loss_value = seg_loss_value


                        # # --- 3. (可选) 训练中可视化 ---
                        # if should_visualize:
                        #     with torch.cuda.amp.autocast(enabled=False):
                        #         with torch.no_grad():
                        #             real_head = seg_head.module if isinstance(seg_head, DistributedDataParallel) else seg_head
                        #             real_head.eval() 
                        #             # first_param = next(real_head.parameters())
                        #             # logger.info(f"【诊断】SegHead Weight dtype: {first_param.dtype}")
                        #             # logger.info(f"【诊断】Neck Out (Before Force Cast) dtype: {neck_out.dtype if isinstance(neck_out, torch.Tensor) else neck_out[0].dtype}")

                        #             if isinstance(neck_out, (list, tuple)):
                        #                 neck_out_fp32 = [x.float() for x in neck_out]
                        #             else:
                        #                 neck_out_fp32 = neck_out.float()

                        #             # logger.info(f"【诊断】Neck Out (After Force Cast) dtype: {neck_out_fp32[0].dtype if isinstance(neck_out_fp32, list) else neck_out_fp32.dtype}")
                        #             pred_result = real_head(
                        #                 inputs=neck_out_fp32,
                        #                 input_query=None
                        #             )
                        #             save_training_visualization(
                        #                 pred_results=pred_result,
                        #                 vis_meta=vis_meta,
                        #                 output_dir=vis_output_dir,
                        #                 epoch=epoch,
                        #                 itr=itr
                        #             )

                        #             real_head.train()
                    else:
                        loss = loss


                # ==================== Backward ====================
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # 梯度裁剪（可选）
                    # torch.nn.utils.clip_grad_norm_(student_perceiver.parameters(), max_norm=1.0)
                    if seg_head is not None:
                        torch.nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=1.0)
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(student_perceiver.parameters(), max_norm=1.0)
                    if seg_head is not None:
                        torch.nn.utils.clip_grad_norm_(seg_head.parameters(), max_norm=1.0)
                
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                # EMA更新teacher perceiver
                # if perceiver_ema:
                #     assert student_perceiver_train, "perceiver_ema=True requires student_perceiver_train=True"
                #     update_teacher_perceiver()

                # EMA更新teacher encoder (使用动态 momentum)
                if encoder_ema:
                    assert encoder_train, "encoder_ema=True requires encoder_train=True"
                    m = next(momentum_scheduler)  # 获取当前 momentum
                    update_teacher_encoder(m)

                # ========== 可视化（在autocast外部）==========
                if should_visualize and valid_samples > 0:
                    with torch.no_grad():
                        real_head = seg_head.module if isinstance(seg_head, DistributedDataParallel) else seg_head
                        real_head.eval()
                        pred_result = real_head(inputs=[x.float() for x in neck_out], input_query=None)
                        save_training_visualization(pred_result, vis_meta, vis_output_dir, epoch, itr)
                        real_head.train()
                return (float(loss), float(jloss), float(sloss), 
                       float(seg_loss_value), float(mask_loss_value), float(traj_loss),
                       float(dice_loss_value), _new_lr, _new_wd)
            (
                loss, jloss, sloss,
                seg_loss_value, mask_loss_value, traj_loss,dice_loss_value,
                _new_lr, _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            jloss_meter.update(jloss)
            sloss_meter.update(sloss)
            seg_loss_meter.update(seg_loss_value)  # 新增
            mask_loss_meter.update(mask_loss_value)
            dice_loss_meter.update(dice_loss_value)
            traj_loss_meter.update(traj_loss) # planner
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            # ==================== Logging ====================
            def log_stats():
                csv_logger.log(
                    epoch + 1, 
                    itr, 
                    loss, 
                    seg_loss_value,  # 新增
                    iter_elapsed_time_ms, 
                    gpu_etime_ms, 
                    data_elapsed_time_ms
                )
                
                # TensorBoard logging
                if rank == 0 and tb_writer is not None:
                    global_step = epoch * ipe + itr
                    
                    # 损失指标
                    tb_writer.add_scalar('Loss/total', loss_meter.avg, global_step)
                    tb_writer.add_scalar('Loss/joint_embedding', jloss_meter.avg, global_step)
                    tb_writer.add_scalar('Loss/autoregressive', sloss_meter.avg, global_step)
                    tb_writer.add_scalar('Loss/segmentation', seg_loss_meter.avg, global_step)  # 新增
                    tb_writer.add_scalar('Loss/segmentation_mask', mask_loss_meter.avg, global_step)
                    tb_writer.add_scalar('Loss/segmentation_dice', dice_loss_meter.avg, global_step)
                    tb_writer.add_scalar('Loss/trajectory', traj_loss_meter.avg,global_step) # planner
                    # 优化器参数
                    tb_writer.add_scalar('Optimization/learning_rate', _new_lr, global_step)
                    tb_writer.add_scalar('Optimization/weight_decay', _new_wd, global_step)
                    
                    # 性能指标
                    tb_writer.add_scalar('Performance/iter_time_ms', iter_time_meter.avg, global_step)
                    tb_writer.add_scalar('Performance/gpu_time_ms', gpu_time_meter.avg, global_step)
                    tb_writer.add_scalar('Performance/data_load_time_ms', data_elapsed_time_meter.avg, global_step)
                    
                    # 内存使用
                    tb_writer.add_scalar('System/cuda_memory_allocated_MB', 
                                        torch.cuda.max_memory_allocated() / 1024.0**2, 
                                        global_step)
                
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "主人，[%d, %5d] loss: %.3f "
                        "[jepa: %.2f+%.2f, seg: %.3f (mask: %.3f, dice: %.3f),traj_loss:%.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1, itr,
                            loss_meter.avg,
                            jloss_meter.avg, sloss_meter.avg,
                            seg_loss_meter.avg, mask_loss_meter.avg, dice_loss_meter.avg,
                            traj_loss_meter.avg,
                            _new_wd, _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg, gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "主人，损失为nan"

        # ==================== Epoch结束 ====================
        logger.info(f"主人，Epoch {epoch + 1} 平均损失: %.3f (JEPA: %.3f, Seg: %.3f)" % 
                   (loss_meter.avg, jloss_meter.avg + sloss_meter.avg, seg_loss_meter.avg))
        
        if rank == 0 and tb_writer is not None:
            tb_writer.add_scalar('Epoch/avg_loss', loss_meter.avg, epoch + 1)
            tb_writer.add_scalar('Epoch/avg_jloss', jloss_meter.avg, epoch + 1)
            tb_writer.add_scalar('Epoch/avg_sloss', sloss_meter.avg, epoch + 1)
            tb_writer.add_scalar('Epoch/avg_seg_loss', seg_loss_meter.avg, epoch + 1)  # 新增
            tb_writer.flush()

        # ==================== 保存Checkpoint ====================
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        # ==================== 验证 ====================
        if val_loader is not None and (epoch + 1) % val_freq == 0:
            logger.info(f"Running validation at epoch {epoch + 1}...")

            # 运行验证
            val_metrics = run_validation(
                encoder=encoder,
                predictor=predictor,
                planner=planner,
                val_loader=val_loader,
                val_sampler=val_sampler,
                config=args,
                epoch=epoch + 1,
                rank=rank,
                world_size=world_size,
                use_tubelet_repeat=False,  # train.py 不使用 tubelet repeat
            )

            # 记录验证指标到 TensorBoard
            if rank == 0 and tb_writer is not None:
                tb_writer.add_scalar('Validation/ADE', val_metrics['ade'], epoch + 1)
                tb_writer.add_scalar('Validation/FDE', val_metrics['fde'], epoch + 1)
                tb_writer.flush()
                logger.info(f"Validation metrics logged to TensorBoard: ADE={val_metrics['ade']:.4f}, FDE={val_metrics['fde']:.4f}")
    
    # ==================== 训练结束 ====================
    if rank == 0 and tb_writer is not None:
        tb_writer.close()
        logger.info("主人，TensorBoard writer已关闭。")
    
    logger.info("主人，训练完成！")


# if __name__ == "__main__":
#     import argparse
#     import yaml
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help='主人，配置文件路径')
#     args = parser.parse_args()
    
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     main(config)

