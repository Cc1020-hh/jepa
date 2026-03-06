# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Z_AR to Trajectory Branch: Connects V-JEPA predictor's autoregressive output (z_ar)
to Drive-JEPA-style TrajectoryHead for trajectory prediction.

Pipeline (Drive-JEPA style):
  z_ar -> Decoder -> keyval_proj  =>  keyval [B, N, d_model]  (KEYVAL)
  query = learnable _query_embedding [num_poses, d_model]     (QUERY)
  query_out = TransformerDecoder(tgt=query, memory=keyval)  # query cross-attends to z_ar
  trajectory = TrajectoryHead(query_out)
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class TrajectoryHead(nn.Module):
    """Drive-JEPA-style trajectory prediction head. Predicts (x, y, θ) per pose."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, 3),  # (x, y, heading)
        )

    def forward(self, object_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, 3)
        poses[..., 2] = poses[..., 2].tanh() * np.pi  # heading in [-pi, pi]
        return {"trajectory": poses}


class ZArTrajectoryBranch(nn.Module):
    """
    Connects z_ar (predictor autoregressive output) to TrajectoryHead.

    z_ar as KEYVAL: query (learnable) cross-attends to z_ar -> TrajectoryHead

    Components:
    - Decoder: Pools z_ar over spatial tokens -> [B, n_steps, D]
    - keyval_proj: Projects to d_model for keyval
    - _query_embedding: Learnable pose queries [num_poses, d_model]
    - Transformer: query attends to keyval (z_ar)
    - TrajectoryHead: MLP on query_out -> [B, num_poses, 3]
    """

    def __init__(
        self,
        z_ar_dim: int,
        tokens_per_frame: int,
        num_poses: int = 8,
        auto_steps: int = 2,
        d_model: int = 256,
        d_ffn: int = 1024,
        tf_num_layers: int = 3,
        tf_num_head: int = 8,
        tf_dropout: float = 0.0,
        decoder_type: str = "spatial_pool",
    ):
        super().__init__()
        self.z_ar_dim = z_ar_dim
        self.tokens_per_frame = tokens_per_frame
        self.num_poses = num_poses
        self.auto_steps = auto_steps
        self.d_model = d_model

        # Decoder: z_ar [B, N, D] -> keyval [B, n_keyval, D] (n_keyval varies by type)
        if decoder_type == "spatial_pool":
            self.decoder = SpatialPoolDecoder(tokens_per_frame, z_ar_dim)
        elif decoder_type == "conv":
            h = w = int(tokens_per_frame**0.5)
            self.decoder = ConvSpatialDecoder(h, w, z_ar_dim)
        elif decoder_type == "attention_pool":
            self.decoder = AttentionPoolDecoder(tokens_per_frame, z_ar_dim)
        elif decoder_type == "transformer_frame":
            self.decoder = TransformerFrameDecoder(
                tokens_per_frame, z_ar_dim, d_model, tf_num_head, tf_dropout
            )
        elif decoder_type == "no_pool":
            self.decoder = NoPoolDecoder(tokens_per_frame, z_ar_dim)
        else:
            self.decoder = SpatialPoolDecoder(tokens_per_frame, z_ar_dim)

        # Project decoded z_ar to d_model for keyval
        self.keyval_proj = nn.Linear(z_ar_dim, d_model)
        # Positional embedding for keyval
        max_keyval = auto_steps * tokens_per_frame if decoder_type == "no_pool" else 64
        self._keyval_embedding = nn.Embedding(max_keyval, d_model)

        # Learnable pose queries (like Drive-JEPA)
        self._query_embedding = nn.Embedding(num_poses, d_model)

        # TransformerDecoder: query cross-attends to keyval (z_ar)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=tf_num_head,
            dim_feedforward=d_ffn,
            dropout=tf_dropout,
            batch_first=True,
        )
        self._transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=tf_num_layers,
        )

        self.trajectory_head = TrajectoryHead(num_poses, d_ffn, d_model)

    def forward(
        self,
        z_ar: torch.Tensor,
        tokens_per_frame: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_ar: [B, N_tokens, D] - predictor autoregressive output (KEYVAL source)
            tokens_per_frame: override if different from init

        Returns:
            {"trajectory": [B, num_poses, 3]}
        """
        tpf = tokens_per_frame or self.tokens_per_frame
        B = z_ar.shape[0]

        # z_ar -> keyval [B, n_steps, d_model]
        decoded = self.decoder(z_ar, tpf)  # [B, n_steps, D]
        keyval = self.keyval_proj(decoded)  # [B, n_steps, d_model]
        n_keyval = keyval.size(1)
        keyval = keyval + self._keyval_embedding.weight[:n_keyval][None, ...]

        # query: learnable [B, num_poses, d_model]
        query = self._query_embedding.weight[None, ...].repeat(B, 1, 1)

        # TransformerDecoder: query (tgt) cross-attends to keyval (memory)
        query_out = self._transformer_decoder(tgt=query, memory=keyval)  # [B, num_poses, d_model]

        return self.trajectory_head(query_out)


class SpatialPoolDecoder(nn.Module):
    """Mean pool over spatial tokens per frame. Simple, loses spatial structure."""

    def __init__(self, tokens_per_frame: int, dim: int):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame

    def forward(self, z_ar: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        B, N, D = z_ar.shape
        n_steps = N // tokens_per_frame
        z_ar = z_ar.view(B, n_steps, tokens_per_frame, D)
        return z_ar.mean(dim=2)  # [B, n_steps, D]


class AttentionPoolDecoder(nn.Module):
    """
    Learnable query attends to frame tokens -> weighted aggregation.
    Preserves important spatial regions via attention weights.
    """

    def __init__(self, tokens_per_frame: int, dim: int):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, z_ar: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        B, N, D = z_ar.shape
        n_steps = N // tokens_per_frame
        z = z_ar.view(B, n_steps, tokens_per_frame, D)  # [B, T, H*W, D]
        q = self.query.expand(B, n_steps, -1, -1)  # [B, T, 1, D]
        attn = torch.einsum("btqd,btnd->btqn", q, z) / (D**0.5)  # [B, T, 1, H*W]
        attn = attn.softmax(dim=-1)
        out = torch.einsum("btqn,btnd->btqd", attn, z).squeeze(2)  # [B, T, D]
        return out


class TransformerFrameDecoder(nn.Module):
    """
    Small transformer within each frame to model spatial relations, then pool.
    Captures token interactions before aggregation.
    """

    def __init__(
        self,
        tokens_per_frame: int,
        dim: int,
        d_model: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.proj_in = nn.Linear(dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pool_token, std=0.02)
        self.proj_out = nn.Linear(d_model, dim)  # back to z_ar_dim for keyval_proj

    def forward(self, z_ar: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        B, N, D = z_ar.shape
        n_steps = N // tokens_per_frame
        z = z_ar.view(B, n_steps, tokens_per_frame, D)
        # Batch: [B*n_steps, H*W, d_model]
        z_flat = z.reshape(B * n_steps, tokens_per_frame, D)
        frame = self.proj_in(z_flat)
        pool_tok = self.pool_token.expand(B * n_steps, -1, -1)
        frame_with_pool = torch.cat([pool_tok, frame], dim=1)
        encoded = self.encoder(frame_with_pool)
        pooled = self.proj_out(encoded[:, 0])  # [B*n_steps, D]
        return pooled.view(B, n_steps, D)


class NoPoolDecoder(nn.Module):
    """
    No spatial aggregation - use full z_ar as keyval.
    Query attends to all tokens. Most expressive, higher compute.
    """

    def __init__(self, tokens_per_frame: int, dim: int):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame

    def forward(self, z_ar: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        return z_ar  # [B, N, D] - no change


class ConvSpatialDecoder(nn.Module):
    """Use conv to aggregate spatial structure (when H, W known)."""

    def __init__(self, h: int, w: int, dim: int):
        super().__init__()
        self.h, self.w = h, w
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, z_ar: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
        B, N, D = z_ar.shape
        n_steps = N // tokens_per_frame
        h = w = int(tokens_per_frame ** 0.5)
        z = z_ar.view(B, n_steps, h, w, D).permute(0, 1, 4, 2, 3)  # [B, T, D, H, W]
        z = z.reshape(B * n_steps, D, h, w)
        z = self.conv(z).flatten(1)  # [B*T, D]
        return z.view(B, n_steps, D)


