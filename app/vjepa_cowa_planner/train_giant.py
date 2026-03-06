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
from app.vjepa_droid.utils import init_opt, init_opt_resample_world_model,init_opt_no_resample_world_model, load_checkpoint, load_pretrained, load_pretrained_safetensors,init_predictor_model
from app.vjepa.utils import init_video_model as init_video_model_vjepa
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from app.vjepa_cowa.RopeResample import RoPEPerceiverResampler
from torch.utils.tensorboard import SummaryWriter
from app.vjepa_cowa.seg_neck2 import SFP
from app.vjepa_cowa.seg_head2 import SimpleSemanticSegHead
# from ddddetection_torchcv.torchcv.modeling.head.semantic_seg_head import SimpleSemanticSegHead
from  app.vjepa_cowa.co_detr_decoder import CoDetrDecoder
from Drive_JEPA.navsim_v1.navsim.agents.drive_jepa_perception_free.drive_jepa_model import TrajectoryHead
# ==================== 新增：评估模块导入 ====================
from app.vjepa_cowa_planner.val_giant import run_validation
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


# ==================== 时序预测 Planner ====================
class MultiModalTemporalPlanner(nn.Module):
    """
    多模态时序轨迹预测器 (Multi-Trajectory Regression + Winner-Takes-All)

    输出 K 条轨迹 + K 个置信度 logit。
    损失计算在外部 wta_loss() 中完成，forward 只负责推理。
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
        num_time_steps: int = 7,
        status_dim: int = 8,
        use_spatial_tokens: bool = False,
        use_step_pooled_tokens: bool = True,
        use_time_aligned_bias: bool = True,
        time_aligned_bias_strength: float = 1.0,
        time_aligned_bias_scope: str = "all_tokens",
        num_modes: int = 6,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.tf_d_model = tf_d_model
        self.tokens_per_frame = tokens_per_frame
        self.num_poses = num_poses
        self.num_time_steps = num_time_steps
        self.use_spatial_tokens = use_spatial_tokens
        self.use_step_pooled_tokens = use_step_pooled_tokens
        self.use_time_aligned_bias = use_time_aligned_bias
        self.time_aligned_bias_strength = time_aligned_bias_strength
        self.time_aligned_bias_scope = time_aligned_bias_scope
        self.num_modes = num_modes
        assert self.time_aligned_bias_scope in ("all_tokens", "pooled_only"), (
            f"Invalid time_aligned_bias_scope={self.time_aligned_bias_scope}"
        )

        # ── 特征投影 ─────────────────────────────────────────────────
        assert self.use_spatial_tokens or self.use_step_pooled_tokens, (
            "Planner must use at least one memory source: spatial tokens or step pooled tokens."
        )
        self.temporal_fc_spatial = nn.Linear(encoder_dim, tf_d_model) if self.use_spatial_tokens else None
        self.temporal_fc_pooled = (
            nn.Linear(encoder_dim * tokens_per_frame, tf_d_model) if self.use_step_pooled_tokens else None
        )

        # ── Positional encoding ───────────────────────────────────────
        # memory tokens + 1(status)
        num_kv = (
            (num_time_steps if self.use_step_pooled_tokens else 0)
            + (tokens_per_frame * num_time_steps if self.use_spatial_tokens else 0)
            + 1
        )
        self.temporal_embedding = nn.Embedding(num_kv, tf_d_model)

        # ── 状态编码 ──────────────────────────────────────────────────
        self.status_encoding = nn.Linear(status_dim, tf_d_model)

        # ── Transformer ───────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=tf_d_model,
            nhead=tf_num_head,
            num_encoder_layers=tf_num_layers,
            num_decoder_layers=tf_num_layers,
            dim_feedforward=tf_d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )

        # ── 多模态 Query：K 个 mode，每个 mode num_poses 个 token ────
        self.query_embedding = nn.Embedding(num_modes * num_poses, tf_d_model)
        query_step_idx = torch.arange(num_poses, dtype=torch.long).repeat(num_modes)
        self.register_buffer("query_step_idx", query_step_idx, persistent=False)

        # ── 轨迹回归头：K 个独立 MLP ──────────────────────────────────
        self.trajectory_heads = nn.ModuleList([
            TrajectoryHead(num_poses, tf_d_ffn, tf_d_model)
            for _ in range(num_modes)
        ])

        # ── 置信度头：输入 [B, K * d_model]，输出 [B, K] ─────────────
        self.confidence_head = nn.Sequential(
            nn.Linear(num_modes * tf_d_model, tf_d_ffn),
            nn.ReLU(inplace=True),
            nn.Linear(tf_d_ffn, num_modes),
        )

    # ─────────────────────────────────────────────────────────────────
    def _build_memory(
        self,
        z_ar: torch.Tensor,
        status_feature: torch.Tensor,
    ) -> tuple:
        """构建 Transformer 的 memory（key-value）序列。"""
        B = z_ar.shape[0]
        T = self.num_time_steps
        P = self.tokens_per_frame
        D = self.encoder_dim

        expected_tokens = T * P
        assert z_ar.ndim == 3, f"Expected z_ar shape [B, N, D], got ndim={z_ar.ndim}"
        assert z_ar.shape[1] == expected_tokens, (
            f"Planner memory reshape mismatch: got N={z_ar.shape[1]}, "
            f"expected num_time_steps*tokens_per_frame={T}*{P}={expected_tokens}"
        )
        assert z_ar.shape[2] == D, (
            f"Planner channel mismatch: got D={z_ar.shape[2]}, expected encoder_dim={D}"
        )
        z_ar_reshaped = z_ar.view(B, T, P, D)

        feat_parts = []
        step_parts = []

        if self.use_step_pooled_tokens:
            # [B, T, P, D] -> [B, T, P*D] -> [B, T, d]
            feat_pooled = self.temporal_fc_pooled(z_ar_reshaped.reshape(B, T, P * D))
            feat_parts.append(feat_pooled)
            step_parts.append(torch.arange(T, device=z_ar.device, dtype=torch.long))

        if self.use_spatial_tokens:
            # [B, T, P, D] -> [B, T, P, d] -> [B, T*P, d]
            feat_spatial = self.temporal_fc_spatial(z_ar_reshaped).reshape(B, T * P, -1)
            feat_parts.append(feat_spatial)
            if self.time_aligned_bias_scope == "pooled_only":
                spatial_steps = torch.full((T * P,), -1, device=z_ar.device, dtype=torch.long)
            else:
                spatial_steps = torch.arange(T, device=z_ar.device, dtype=torch.long).repeat_interleave(P)
            step_parts.append(spatial_steps)

        feat = torch.cat(feat_parts, dim=1)
        memory_step_idx = torch.cat(step_parts, dim=0)
        T_feat = feat.shape[1]

        # Temporal positional encoding
        feat = feat + self.temporal_embedding.weight[:T_feat].unsqueeze(0)

        # Status token
        status = self.status_encoding(status_feature).unsqueeze(1)          # [B,1,d]
        status = status + self.temporal_embedding.weight[T_feat:T_feat + 1].unsqueeze(0)

        # -1 表示 status token，不参与时间距离惩罚
        status_step_idx = torch.full((1,), -1, device=z_ar.device, dtype=torch.long)
        memory_step_idx = torch.cat([memory_step_idx, status_step_idx], dim=0)

        return torch.cat([feat, status], dim=1), memory_step_idx   # [B, M, d], [M]

    def _build_time_aligned_memory_bias(
        self,
        query_step_idx: torch.Tensor,
        memory_step_idx: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """构建时间对齐 bias（用于 decoder cross-attn 的 memory_mask）。"""
        Q = query_step_idx.shape[0]
        M = memory_step_idx.shape[0]
        if (not self.use_time_aligned_bias) or self.time_aligned_bias_strength <= 0:
            return torch.zeros(Q, M, device=query_step_idx.device, dtype=dtype)

        q = query_step_idx.to(torch.float32).unsqueeze(1)  # [Q,1]
        m = memory_step_idx.to(torch.float32).unsqueeze(0)  # [1,M]
        distance = (q - m).abs()

        # 避免不同 T 下惩罚尺度失衡
        norm = max(1.0, float(self.num_time_steps - 1))
        bias = -self.time_aligned_bias_strength * (distance / norm)

        # status token(step=-1)不施加距离惩罚
        status_mask = memory_step_idx.eq(-1).unsqueeze(0)  # [1,M]
        bias = torch.where(status_mask, torch.zeros_like(bias), bias)
        return bias.to(dtype=dtype)

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        z_ar: torch.Tensor,
        status_feature: torch.Tensor,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
            "trajectories" : [B, K, num_poses, 3]   (x, y, yaw)
            "confidences"  : [B, K]                  unnormalized logits
        """
        B = z_ar.shape[0]
        K = self.num_modes

        # 1. Memory
        memory, memory_step_idx = self._build_memory(z_ar, status_feature)           # [B, M, d], [M]

        # 2. Multi-modal queries
        query = self.query_embedding.weight.unsqueeze(0).expand(B, -1, -1)
        memory_bias = self._build_time_aligned_memory_bias(
            self.query_step_idx,
            memory_step_idx,
            dtype=memory.dtype,
        )
        query_out = self.transformer(src=memory, tgt=query, memory_mask=memory_bias)          # [B, K*num_poses, d]
        query_out = query_out.view(B, K, self.num_poses, self.tf_d_model)

        # 3. Per-mode trajectory regression
        # TrajectoryHead 可能返回 dict {"trajectory": Tensor} 或直接返回 Tensor
        traj_list = []
        for k in range(K):
            head_out = self.trajectory_heads[k](query_out[:, k])
            if isinstance(head_out, dict):
                traj_k = head_out["trajectory"]                      # [B, num_poses, 3]
            else:
                traj_k = head_out                                    # [B, num_poses, 3]
            traj_list.append(traj_k)

        trajs = torch.stack(traj_list, dim=1)                        # [B, K, num_poses, 3]

        # 4. Confidence (pool over poses dim → [B, K, d] → [B, K*d] → [B, K])
        conf_feat = query_out.mean(dim=2)                            # [B, K, d]
        conf_logits = self.confidence_head(
            conf_feat.reshape(B, K * self.tf_d_model)
        )                                                             # [B, K]

        return {
            "trajectories": trajs,
            "confidences": conf_logits,
        }


def wta_loss(
    pred_trajs: torch.Tensor,
    pred_conf_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    conf_loss_weight: float = 1.0,
    alpha: float = 5.0,
    eps: float = 1e-6,
) -> dict:
    """
    Winner-Takes-All 多模态轨迹损失 (原版 - 硬标签)

    Parameters
    ----------
    pred_trajs      : [B, K, num_poses, 3]  预测轨迹（K 条）
    pred_conf_logits: [B, K]                置信度 logit（未经 softmax）
    gt_traj         : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight : 回归损失权重
    conf_loss_weight: 置信度损失权重
    alpha           : 长度归一化系数
    eps             : 数值稳定

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : WTA 回归损失（scalar）
        "conf_loss" : 置信度 CE 损失（scalar）
        "cover_loss": Cover损失（原版为0）
        "winner_idx": [B] 每个样本的 winner mode 下标（用于 logging）
    """
    B, K, num_poses, _ = pred_trajs.shape

    # ── Step 1: 找 winner ─────────────────────────────────────────────
    # 计算每条预测轨迹与 GT 的 ADE（平均位移误差，只用 xy）
    gt_expanded = gt_traj.unsqueeze(1).expand_as(pred_trajs)        # [B, K, num_poses, 3]

    # ADE: 只用 xy 两维
    dist_xy = torch.norm(
        pred_trajs[..., :2] - gt_expanded[..., :2],
        dim=-1,
    ).mean(dim=-1)                                                   # [B, K]

    winner_idx = dist_xy.argmin(dim=1)                              # [B]

    # ── Step 2: WTA 回归损失（只对 winner 计算）─────────────────────
    # 取出 winner 轨迹
    winner_idx_exp = winner_idx.view(B, 1, 1, 1).expand(B, 1, num_poses, 3)
    winner_traj = pred_trajs.gather(dim=1, index=winner_idx_exp).squeeze(1)
    # winner_traj: [B, num_poses, 3]

    # 长度归一化 L1 损失
    per_sample_l1 = (winner_traj - gt_traj).abs().mean(dim=[1, 2])  # [B]
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)             # [B]
    w = 1.0 / (alpha + arc_len)
    w = w * (w.numel() / (w.sum() + eps))
    reg_loss = (w * per_sample_l1).mean()

    # ── Step 3: 置信度损失（CE，winner 类别监督）──────────────────────
    conf_loss = F.cross_entropy(
        pred_conf_logits,
        winner_idx,
        reduction="mean",
    )

    # ── Step 4: 合并 ──────────────────────────────────────────────────
    total = reg_loss_weight * reg_loss + conf_loss_weight * conf_loss

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": conf_loss,
        "cover_loss": torch.tensor(0.0, device=pred_trajs.device),  # 原版无cover损失
        "winner_idx": winner_idx,
    }


def wta_loss_v2(
    pred_trajs: torch.Tensor,
    pred_conf_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    conf_loss_weight: float = 1.0,
    cover_loss_weight: float = 0.1,
    alpha: float = 5.0,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> dict:
    """
    Winner-Takes-All 多模态轨迹损失 (改进版 - 软标签 + Cover损失)

    改进点:
    1. 使用软标签代替硬标签，提高泛化性
    2. 添加Cover损失，鼓励不同mode覆盖不同的轨迹空间

    Parameters
    ----------
    pred_trajs      : [B, K, num_poses, 3]  预测轨迹（K 条）
    pred_conf_logits: [B, K]                置信度 logit（未经 softmax）
    gt_traj         : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight : 回归损失权重
    conf_loss_weight: 置信度损失权重
    cover_loss_weight: Cover损失权重（鼓励轨迹多样性）
    alpha           : 长度归一化系数
    temperature     : 软标签温度参数，越大越平滑
    eps             : 数值稳定

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : WTA 回归损失（scalar）
        "conf_loss" : 置信度损失（scalar）
        "cover_loss": Cover损失（scalar）
        "winner_idx": [B] 每个样本的 winner mode 下标
    """
    B, K, num_poses, _ = pred_trajs.shape

    # ── Step 1: 计算所有轨迹与GT的距离 ───────────────────────────────
    gt_expanded = gt_traj.unsqueeze(1).expand_as(pred_trajs)        # [B, K, num_poses, 3]

    # ADE: 只用 xy 两维
    dist_xy = torch.norm(
        pred_trajs[..., :2] - gt_expanded[..., :2],
        dim=-1,
    ).mean(dim=-1)                                                   # [B, K]

    # ── Step 2: Winner选择 ───────────────────────────────────────────
    winner_idx = dist_xy.argmin(dim=1)                              # [B]

    # ── Step 3: 回归损失（只对winner计算）────────────────────────────
    winner_idx_exp = winner_idx.view(B, 1, 1, 1).expand(B, 1, num_poses, 3)
    winner_traj = pred_trajs.gather(dim=1, index=winner_idx_exp).squeeze(1)

    # 长度归一化 L1 损失
    per_sample_l1 = (winner_traj - gt_traj).abs().mean(dim=[1, 2])  # [B]
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)             # [B]
    w = 1.0 / (alpha + arc_len)
    w = w * (w.numel() / (w.sum() + eps))
    reg_loss = (w * per_sample_l1).mean()

    # ── Step 4: 软标签置信度损失 ─────────────────────────────────────
    # 使用距离的softmax作为软标签，距离越近权重越高
    # -dist_xy/temperature: 距离越小，softmax值越大
    soft_target = F.softmax(-dist_xy / temperature, dim=1)          # [B, K]

    # 使用交叉熵损失（pred_conf_logits是logits，soft_target是概率分布）
    log_probs = F.log_softmax(pred_conf_logits, dim=1)              # [B, K]
    conf_loss = -(soft_target * log_probs).sum(dim=1).mean()        # scalar

    # ── Step 5: Cover损失 - 鼓励轨迹多样性 ─────────────────────────────
    if K > 1:
        # 计算不同mode之间的轨迹相似度
        # [B, K, num_poses, 3] -> [B, K, num_poses*3]
        traj_flat = pred_trajs.flatten(2)                           # [B, K, num_poses*3]

        # L2归一化
        traj_norm = F.normalize(traj_flat, p=2, dim=-1)             # [B, K, num_poses*3]

        # 计算余弦相似度矩阵 [B, K, K]
        sim_matrix = torch.bmm(traj_norm, traj_norm.transpose(1, 2))

        # 移除对角线（自相似度），只考虑不同mode之间的相似度
        mask = 1.0 - torch.eye(K, device=pred_trajs.device).unsqueeze(0)  # [1, K, K]
        off_diag_sim = sim_matrix * mask                            # [B, K, K]

        # Cover损失：惩罚过高的相似度
        # 平方操作放大高相似度的惩罚
        cover_loss = (off_diag_sim ** 2).sum(dim=[1, 2]) / (K * (K - 1))  # [B]
        cover_loss = cover_loss.mean()
    else:
        cover_loss = torch.tensor(0.0, device=pred_trajs.device)

    # ── Step 6: 合并总损失 ───────────────────────────────────────────
    total = (reg_loss_weight * reg_loss +
             conf_loss_weight * conf_loss +
             cover_loss_weight * cover_loss)

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": conf_loss,
        "cover_loss": cover_loss,
        "winner_idx": winner_idx,
    }
def select_best_trajectory(
    pred_trajs: torch.Tensor,
    conf_logits: torch.Tensor,
) -> torch.Tensor:
    """
    推理时按最高置信度选出最优轨迹。

    Parameters
    ----------
    pred_trajs  : [B, K, num_poses, 3]
    conf_logits : [B, K]

    Returns
    -------
    best_traj   : [B, num_poses, 3]
    """
    best_idx = conf_logits.argmax(dim=1)                            # [B]
    best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(
        -1, 1, pred_trajs.shape[2], pred_trajs.shape[3]
    )
    return pred_trajs.gather(1, best_idx_exp).squeeze(1)            # [B, num_poses, 3]


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


def prepare_status_feature(states, actions, mode: str = "last"):
    """从 states 和 actions 提取状态特征。

    mode:
        - "last": 使用最后一个时刻（历史行为兼容，但可能存在未来信息泄漏风险）
        - "first": 使用起始时刻（严格因果、更保守）
        - "history_pool": 对时序做均值池化（折中方案）
    """
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
        velocity = states[:, -1, 6:7]  # [B, 1]
        if states.shape[1] >= 2:
            acceleration = states[:, -1, 6:7] - states[:, -2, 6:7]
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, -1, 5:6]  # [B, 1]
        xy = states[:, -1, 0:2]  # [B, 2]
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, -1, :3]
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)

    return torch.cat([velocity, acceleration, yaw, xy, action_feat], dim=-1)  # [B, 8]


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
    # 新增：包含 encoder, predictor, seg 的完整预训练 checkpoint
    p_file_full = cfgs_meta.get("pretrain_checkpoint_full", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    load_encoder = cfgs_meta.get("load_encoder", True)
    load_seg = cfgs_meta.get("load_seg", True)  # 新增：是否加载 seg 权重
    load_planner = cfgs_meta.get("load_planner", True)  # 新增：是否加载 planner 权重
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
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
    use_mask_tokens = cfgs_model.get("use_mask_tokens", False)
    zero_init_mask_tokens = cfgs_model.get("zero_init_mask_tokens", True)
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
    # EMA 动态 momentum 范围: [start, end], 从小到大逐渐增加
    ema_start = cfgs_ema.get("ema_start", 0.996)  # 默认起始值
    ema_end = cfgs_ema.get("ema_end", 0.999)      # 默认结束值
    ema = [ema_start, ema_end]  # 与 train.py 保持一致的格式
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
    # -- PLANNER (时序预测)
    cfgs_planner = args.get("planner", {})
    use_planner = cfgs_planner.get("use_planner", True)  # 是否使用 planner
    status_mode = cfgs_planner.get("status_mode", "last")
    tf_d_model = cfgs_planner.get("tf_d_model", 256)
    tf_d_ffn = cfgs_planner.get("tf_d_ffn", 1024)
    tf_num_layers = cfgs_planner.get("tf_num_layers", 3)
    tf_num_head = cfgs_planner.get("tf_num_head", 8)
    tf_dropout = cfgs_planner.get("tf_dropout", 0.0)
    planner_loss_weight = cfgs_planner.get("planner_loss_weight", 1.0)
    use_spatial_tokens = cfgs_planner.get("use_spatial_tokens", False)  # 是否保留空间token
    use_step_pooled_tokens = cfgs_planner.get("use_step_pooled_tokens", True)
    use_time_aligned_bias = cfgs_planner.get("use_time_aligned_bias", True)
    time_aligned_bias_strength = cfgs_planner.get("time_aligned_bias_strength", 1.0)
    time_aligned_bias_scope = cfgs_planner.get("time_aligned_bias_scope", "all_tokens")
    z_ar_mode = cfgs_planner.get("z_ar_mode", "full")  # full / first_step
    num_modes = cfgs_planner.get("num_modes", 6)
    conf_loss_weight = cfgs_planner.get("conf_loss_weight", 1.0)
    reg_loss_weight = cfgs_planner.get("reg_loss_weight", 1.0)
    # WTA损失版本选择 (v1: 原版硬标签, v2: 改进版软标签+Cover损失)
    wta_loss_version = cfgs_planner.get("wta_loss_version", "v1")
    wta_temperature = cfgs_planner.get("wta_temperature", 1.0)  # v2专用: 软标签温度
    cover_loss_weight = cfgs_planner.get("cover_loss_weight", 0.1)  # v2专用: Cover损失权重
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
        # 使用 LOCAL_RANK 分配不同的 GPU (torchrun 会自动设置 LOCAL_RANK)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
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

    # -- 初始化模型 (使用 vjepa 的 init_video_model 创建 encoder)
    logger.info("begin init encoder using init_video_model_vjepa:")
    encoder, _ = init_video_model_vjepa(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=10,
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    target_encoder = copy.deepcopy(encoder)
    logger.info("end init encoder")

    # 获取 encoder 的 embed_dim 用于 predictor 初始化
    encoder_embed_dim = encoder.backbone.embed_dim
    logger.info(f"encoder_embed_dim: {encoder_embed_dim}")

    # 初始化 action-conditioned predictor (使用 vjepa_droid 的 init_predictor_model)
    logger.info("begin init action-conditioned predictor:")
    predictor = init_predictor_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        embed_dim=encoder_embed_dim,
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
        target_shape=None
    )
    logger.info("end init predictor")

    # 打印参数量
    encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"init encoder_params: {encoder_params / 1e6:>8.2f}M")
    target_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    print(f"init target_encoder_params: {target_encoder_params / 1e6:>8.2f}M")
    predictor_params = sum(p.numel() for p in predictor.parameters())
    print(f"init predictor_params: {predictor_params / 1e6:>8.2f}M")

    if use_segmentation:
        seg_neck = SFP(
            input_channels = [1408],
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
    # -- PLANNER (时序预测)
    num_poses = (target_frame // tubelet_size) - 1
    assert z_ar_mode in ("full", "first_step"), f"Invalid planner.z_ar_mode={z_ar_mode}"
    num_time_steps = num_poses if z_ar_mode == "full" else 1
    # 获取 encoder 维度 (MultiSeqWrapper 包装后的 encoder 使用 backbone.embed_dim)
    encoder_dim = encoder.backbone.embed_dim

    if use_planner:
        planner = MultiModalTemporalPlanner(
            encoder_dim=encoder_dim,
            tf_d_model=tf_d_model,
            tf_d_ffn=tf_d_ffn,
            tf_num_layers=tf_num_layers,
            tf_num_head=tf_num_head,
            tf_dropout=tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_time_steps,
            status_dim=8,
            use_spatial_tokens=use_spatial_tokens,
            use_step_pooled_tokens=use_step_pooled_tokens,
            use_time_aligned_bias=use_time_aligned_bias,
            time_aligned_bias_strength=time_aligned_bias_strength,
            time_aligned_bias_scope=time_aligned_bias_scope,
            num_modes=num_modes,
        ).to(device)

        planner_params = sum(p.numel() for p in planner.parameters())
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(TemporalPlanner, num_time_steps={num_time_steps}, z_ar_mode={z_ar_mode}, "
            f"use_spatial_tokens={use_spatial_tokens}, use_step_pooled_tokens={use_step_pooled_tokens}, "
            f"use_time_aligned_bias={use_time_aligned_bias}, bias_strength={time_aligned_bias_strength}, "
            f"bias_scope={time_aligned_bias_scope})"
        )
    else:
        planner = None
        logger.info("use_planner=False, planner is disabled")

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
    if use_planner:
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

    if use_planner:
        planner = DistributedDataParallel(planner, find_unused_parameters=False)

    # 冻结参数
    for p in encoder.parameters():
        if encoder_train:
            p.requires_grad = True
        else:
            p.requires_grad = False

    # ==================== 添加 encoder 参数到 optimizer ====================
    # 只有当 encoder_train=True 时才添加
    if encoder_train:
        optimizer.add_param_group({
            "params": encoder.parameters(),
            "lr": lr * enc_lr_scale,  # 使用 encoder 专用的学习率缩放
            "weight_decay": wd,
        })
        logger.info(f"Added encoder parameters to optimizer with lr_scale={enc_lr_scale}")
        # p.requires_grad = False
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
    planner_params_total = sum(p.numel() for p in planner.parameters() if p.requires_grad) if use_planner else 0

    logger.info(f"{'='*50}")
    logger.info(f"Trainable Parameters Summary:")
    logger.info(f"  encoder:           {sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    logger.info(f"  predictor:         {sum(p.numel() for p in predictor.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    if seg_neck is not None:
        logger.info(f"  seg_neck:          {sum(p.numel() for p in seg_neck.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    if seg_head is not None:
        logger.info(f"  seg_head:          {sum(p.numel() for p in seg_head.parameters() if p.requires_grad) / 1e6:>8.2f}M")
    if use_planner:
        logger.info(f"  planner:           {planner_params_total / 1e6:>8.2f}M")
    logger.info(f"{'─'*50}")
    logger.info(f"  ALL trainable:     {total_trainable_params / 1e6:>8.2f}M")
    logger.info(f"{'='*50}")

    # ==================== 加载预训练权重 ====================
    # 辅助函数：加载 state_dict（处理 DDP 的 module. 前缀）
    def load_state_dict(model, state_dict, name):
        """加载 state_dict，自动处理 DDP 的 module. 前缀"""
        model_unwrapped = model.module if hasattr(model, 'module') else model
        # 移除 'module.' 前缀
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        missing, unexpected = model_unwrapped.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Loaded {name}: missing={len(missing)}, unexpected={len(unexpected)}")

    # # 1. 加载 V-JEPA 预训练权重 (safetensors 格式)
    # _, predictor, _ = load_pretrained_safetensors(
    #     r_path=p_file,
    #     predictor=predictor,
    #     load_predictor=load_predictor,
    #     load_encoder=False
    # )

    # 2. 加载完整预训练 checkpoint (.pt 格式，包含 encoder, predictor, seg)
    if p_file_full is not None and os.path.exists(p_file_full):
        logger.info(f"Loading full pretrained checkpoint from {p_file_full}")
        checkpoint = torch.load(p_file_full, map_location='cpu')

        if load_encoder and 'encoder' in checkpoint:
            load_state_dict(encoder, checkpoint['encoder'], 'encoder')
            target_encoder.load_state_dict(encoder.state_dict())
            logger.info("Synchronized target_encoder with encoder")

        # if load_predictor and 'predictor' in checkpoint:
        #     load_state_dict(predictor, checkpoint['predictor'], 'predictor')

        # if load_seg:
        #     if seg_neck is not None and 'seg_neck' in checkpoint:
        #         load_state_dict(seg_neck, checkpoint['seg_neck'], 'seg_neck')
        #     if seg_head is not None and 'seg_head' in checkpoint:
        #         load_state_dict(seg_head, checkpoint['seg_head'], 'seg_head')

        # # ==================== 新增：加载 Planner 权重 ====================
        # if load_planner and 'planner' in checkpoint:
        #     load_state_dict(planner, checkpoint['planner'], 'planner')

        logger.info("Full pretrained checkpoint loaded successfully!")
    elif p_file_full is not None:
        logger.warning(f"Full pretrained checkpoint not found: {p_file_full}")
    # @torch.no_grad()
    # def update_teacher_perceiver():
    #     """使用EMA更新teacher perceiver的参数"""
    #     student_state = student_perceiver.module.state_dict() if hasattr(student_perceiver, 'module') else student_perceiver.state_dict()
    #     teacher_state = teacher_perceiver.state_dict()

    #     for key in student_state:
    #         teacher_state[key] = ema_decay * teacher_state[key] + (1 - ema_decay) * student_state[key]

    #     teacher_perceiver.load_state_dict(teacher_state)

    # -- momentum schedule (动态 EMA，与 train.py 保持一致)
    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs)
        for i in range(int(ipe * num_epochs) + 1)
    )

    @torch.no_grad()
    def update_teacher_encoder(m):
        """
        使用动态 momentum 更新 teacher encoder 的参数
        采用高效的原地操作方式 (与 train.py 一致)

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
        if use_planner and planner is not None:
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
        reg_loss_meter  = AverageMeter()
        conf_loss_meter = AverageMeter()
        cover_loss_meter = AverageMeter()  # v2专用: Cover损失
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

                        # 只取 ::tubelet_size 的帧
                        sampled_clips = target_clips[:, :, ::tubelet_size, :, :]  # [B, C, T//tubelet_size, H, W]
                        num_sampled_frames = sampled_clips.shape[2]

                        # 对每帧复制 tubelet_size 次扩展成视频格式
                        # [B, C, T//tubelet_size, H, W] -> [B, C, T//tubelet_size, tubelet_size, H, W] -> [B, C, T, H, W]
                        sampled_clips = sampled_clips.unsqueeze(3).repeat(1, 1, 1, tubelet_size, 1, 1)
                        sampled_clips = sampled_clips.view(B, C, -1, H, W)  # [B, C, T, H, W]

                        # Encoder (frozen) - 使用 vjepa 方式调用 (输入为 list)
                        # MultiSeqWrapper 期望输入是 list，返回也是 list
                        # 注意：PatchEmbed3D 期望输入格式为 [B, C, T, H, W]，不需要 permute
                        h_target = target_encoder([sampled_clips])
                        h_ = h_target[0]  # 取出第一个结果 [B, N, D]

                        # Teacher Perceiver (frozen, no grad)
                        # h = teacher_perceiver(h_)  # [B, num_latents, embed_dim]

                        if normalize_reps:
                            h_ = F.layer_norm(h_, (h_.size(-1),))

                        return h_

                def forward_context(context_clips):
                    """Student分支：encoder -> perceiver -> 输出"""
                    B, C, T, H, W = context_clips.shape
                    assert C == 3

                    grad_ctx = torch.enable_grad() if encoder_train else torch.no_grad()
                    with grad_ctx:
                        B, C, T, H, W = context_clips.shape
                        assert C == 3, f"Expected 3 channels (RGB), got {C}"
                        assert T % tubelet_size == 0, f"T={T} not divisible by tubelet_size={tubelet_size}"

                        # 只取 ::tubelet_size 的帧
                        sampled_clips = context_clips[:, :, ::tubelet_size, :, :]  # [B, C, T//tubelet_size, H, W]
                        num_sampled_frames = sampled_clips.shape[2]

                        # 对每帧复制 tubelet_size 次扩展成视频格式
                        # [B, C, T//tubelet_size, H, W] -> [B, C, T//tubelet_size, tubelet_size, H, W] -> [B, C, T, H, W]
                        sampled_clips = sampled_clips.unsqueeze(3).repeat(1, 1, 1, tubelet_size, 1, 1)
                        sampled_clips = sampled_clips.view(B, C, -1, H, W)  # [B, C, T, H, W]

                        # Encoder (frozen) - 使用 vjepa 方式调用 (输入为 list)
                        # 注意：PatchEmbed3D 期望输入格式为 [B, C, T, H, W]，不需要 permute
                        z_context = encoder([sampled_clips])
                        z = z_context[0]  # 取出第一个结果 [B, N, D]

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
                    traj_loss = torch.tensor(0.0, device=device)
                    _reg_loss = torch.tensor(0.0, device=device)
                    _conf_loss = torch.tensor(0.0, device=device)
                    _cover_loss = torch.tensor(0.0, device=device)
                    _winner = None

                    if use_planner and planner is not None:
                        B = z_ar.shape[0]
                        z_ar_planner = z_ar if z_ar_mode == "full" else z_ar[:, :tokens_per_frame]

                        # 准备状态特征
                        status_feature = prepare_status_feature(states, actions, mode=status_mode)

                        # Planner forward
                        # ── Planner forward (多模态) ───────────────────────
                        planner_out = planner(z_ar_planner, status_feature)
                        pred_trajs   = planner_out["trajectories"]   # [B, K, num_poses, 3]
                        pred_conf    = planner_out["confidences"]    # [B, K]

                        # ── GT 轨迹构建（与原逻辑相同） ────────────────────
                        StateSE2_indices = [0, 1, 5]
                        states_se2 = states[:, :, StateSE2_indices]
                        origin_x   = states_se2[:, 0, 0]
                        origin_y   = states_se2[:, 0, 1]
                        origin_yaw = states_se2[:, 0, 2]

                        dx   = states_se2[:, 1:, 0] - origin_x[:, None]
                        dy   = states_se2[:, 1:, 1] - origin_y[:, None]
                        dyaw = states_se2[:, 1:, 2] - origin_yaw[:, None]

                        cos_h = torch.cos(-origin_yaw)
                        sin_h = torch.sin(-origin_yaw)
                        ego_x   = cos_h[:, None] * dx - sin_h[:, None] * dy
                        ego_y   = sin_h[:, None] * dx + cos_h[:, None] * dy
                        ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                        gt_trajectory = torch.stack(
                            [ego_x, ego_y, ego_yaw], dim=-1
                        )                                              # [B, T-1, 3]
                        gt_trajectory = gt_trajectory[:, :num_poses]   # [B, num_poses, 3]

                        # ── WTA 损失（支持版本选择）───────────────────────────
                        if wta_loss_version == "v2":
                            wta_result = wta_loss_v2(
                                pred_trajs=pred_trajs,
                                pred_conf_logits=pred_conf,
                                gt_traj=gt_trajectory,
                                reg_loss_weight=reg_loss_weight,
                                conf_loss_weight=conf_loss_weight,
                                cover_loss_weight=cover_loss_weight,
                                alpha=5.0,
                                temperature=wta_temperature,
                            )
                        else:  # v1 (默认)
                            wta_result = wta_loss(
                                pred_trajs=pred_trajs,
                                pred_conf_logits=pred_conf,
                                gt_traj=gt_trajectory,
                                reg_loss_weight=reg_loss_weight,
                                conf_loss_weight=conf_loss_weight,
                                alpha=5.0,
                            )
                        traj_loss  = wta_result["loss"]
                        _reg_loss  = wta_result["reg_loss"]
                        _conf_loss = wta_result["conf_loss"]
                        _cover_loss = wta_result["cover_loss"]  # v1为0, v2有值
                        _winner    = wta_result["winner_idx"]          # [B] for logging

                        # ── 调试日志（前几步） ────────────────────────────
                        if itr < 20:
                            logger.info(
                                f"WTA(v{wta_loss_version}): winner_modes={_winner.tolist()}, "
                                f"reg={_reg_loss.item():.4f}, "
                                f"conf={_conf_loss.item():.4f}, "
                                f"cover={_cover_loss.item():.4f}"
                            )

                        # ── 轨迹可视化（用置信度最高的那条） ─────────────
                        if should_visualize:
                            best_traj = select_best_trajectory(pred_trajs, pred_conf)
                            visualize_trajectory(
                                pred_traj=best_traj,
                                gt_traj=gt_trajectory,
                                output_dir=vis_output_dir,
                                epoch=epoch,
                                itr=itr,
                                limit=5,
                            )

                    # ── 合并进总损失 ───────────────────────────────────
                    loss = jepa_loss + planner_loss_weight * traj_loss
                    # ==================== 新增：分割损失 ====================
                    seg_loss_value = 0.0
                    mask_loss_value = 0.0
                    dice_loss_value = 0.0
                    valid_samples = 0  # 初始化，避免 UnboundLocalError
                    neck_out = None
                    vis_meta = None
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
                       float(dice_loss_value),float(_reg_loss), float(_conf_loss), float(_cover_loss),
                        _new_lr, _new_wd)
            (
                loss, jloss, sloss,
                seg_loss_value, mask_loss_value, traj_loss,dice_loss_value,
                reg_loss_value, conf_loss_value, cover_loss_value,
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
            # 新增 meter
            reg_loss_meter.update(reg_loss_value)
            conf_loss_meter.update(conf_loss_value)
            cover_loss_meter.update(cover_loss_value)
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
                    mask_loss_value,
                    dice_loss_value,
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

                    # Planner详细损失 (v2专用)
                    if wta_loss_version == "v2":
                        tb_writer.add_scalar('Planner/cover_loss', cover_loss_meter.avg, global_step)

                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    # 根据WTA版本显示不同的日志格式
                    if wta_loss_version == "v2":
                        logger.info(
                            "主人，[%d, %5d] loss: %.3f "
                            "[jepa: %.2f+%.2f, seg: %.3f (mask: %.3f, dice: %.3f)] "
                            "[traj=[wta:%.3f reg:%.3f conf:%.3f cover:%.3f]] "
                            "[wd: %.2e] [lr: %.2e] "
                            "[mem: %.2e] "
                            "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                            % (
                                epoch + 1, itr,
                                loss_meter.avg,
                                jloss_meter.avg, sloss_meter.avg,
                                seg_loss_meter.avg, mask_loss_meter.avg, dice_loss_meter.avg,
                                traj_loss_meter.avg, reg_loss_meter.avg, conf_loss_meter.avg, cover_loss_meter.avg,
                                _new_wd, _new_lr,
                                torch.cuda.max_memory_allocated() / 1024.0**2,
                                iter_time_meter.avg, gpu_time_meter.avg,
                                data_elapsed_time_meter.avg,
                            )
                        )
                    else:  # v1
                        logger.info(
                            "主人，[%d, %5d] loss: %.3f "
                            "[jepa: %.2f+%.2f, seg: %.3f (mask: %.3f, dice: %.3f)] "
                            "[traj=[wta:%.3f reg:%.3f conf:%.3f]] "
                            "[wd: %.2e] [lr: %.2e] "
                            "[mem: %.2e] "
                            "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                            % (
                                epoch + 1, itr,
                                loss_meter.avg,
                                jloss_meter.avg, sloss_meter.avg,
                                seg_loss_meter.avg, mask_loss_meter.avg, dice_loss_meter.avg,
                                traj_loss_meter.avg, reg_loss_meter.avg, conf_loss_meter.avg,
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
        if use_planner and val_loader is not None and (epoch + 1) % val_freq == 0:
            logger.info(f"Running validation at epoch {epoch + 1}...")

            # 运行验证 (使用 val_giant.py，逻辑与 train_giant.py 保持一致)
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
            )

            # 记录验证指标到 TensorBoard
            if rank == 0 and tb_writer is not None:
                tb_writer.add_scalar('Validation/ADE', val_metrics['ade'], epoch + 1)
                tb_writer.add_scalar('Validation/FDE', val_metrics['fde'], epoch + 1)
                if 'minade_k' in val_metrics:
                    tb_writer.add_scalar('Validation/minADE@K', val_metrics['minade_k'], epoch + 1)
                if 'minfde_k' in val_metrics:
                    tb_writer.add_scalar('Validation/minFDE@K', val_metrics['minfde_k'], epoch + 1)
                tb_writer.flush()
                msg = f"Validation metrics logged to TensorBoard: ADE={val_metrics['ade']:.4f}, FDE={val_metrics['fde']:.4f}"
                if 'minade_k' in val_metrics and 'minfde_k' in val_metrics:
                    msg += f", minADE@K={val_metrics['minade_k']:.4f}, minFDE@K={val_metrics['minfde_k']:.4f}"
                logger.info(msg)
    
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

