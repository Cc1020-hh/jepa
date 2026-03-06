# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
import warnings

import torch
import yaml

import src.models.predictor as vit_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, LinearDecaySchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper, PredictorMultiSeqWrapper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

MAX_RETRIES = 3


def build_eval_args(
    model_name,
    patch_size,
    tubelet_size,
    num_frames,
    logging_folder,
    checkpoint,
    write_tag,
    eval_cfg_paths,
    uniform_power=False,
    use_sdpa=False,
    clip_duration=None,
    use_silu=False,
    wide_silu=True,
    tag=None,
):
    """
    Helper function to parse the pre-training configs to construct the
    evaluation configs, return as a list of eval configs.
    """
    # By convention, the pre-training config should specify any required evals
    # in the 'evals' key
    if eval_cfg_paths is None:
        logger.info("No evaluations specified!")
        return

    eval_nodes = None
    eval_tasks_per_node = None
    args_eval = []
    for i, f in enumerate(eval_cfg_paths):
        with open(f, "r") as y_file:
            _args = yaml.load(y_file, Loader=yaml.FullLoader)
            _tag = _args.get("tag", "")
            _args["tag"] = f"{tag}-{_tag}"
            _nodes = _args.get("nodes", None)
            _tasks = _args.get("tasks_per_node", 8)
            eval_nodes = _nodes if eval_nodes is None else eval_nodes
            eval_tasks_per_node = _tasks if eval_tasks_per_node is None else eval_tasks_per_node
            if (eval_nodes != _nodes) or (eval_tasks_per_node != _tasks):
                warnings.warn("Configs for online evals must use same number of nodes for slurm-batch processing")

            # Model params
            _args["pretrain"] = {}
            _args["pretrain"]["model_name"] = model_name
            _args["pretrain"]["patch_size"] = patch_size
            _args["pretrain"]["tubelet_size"] = tubelet_size
            _args["pretrain"]["uniform_power"] = uniform_power
            _args["pretrain"]["use_sdpa"] = use_sdpa
            _args["pretrain"]["clip_duration"] = clip_duration
            _args["pretrain"]["use_silu"] = use_silu
            _args["pretrain"]["wide_silu"] = wide_silu

            # Data params
            _args["pretrain"]["frames_per_clip"] = num_frames

            # Misc
            _args["pretrain"]["folder"] = logging_folder
            _args["pretrain"]["checkpoint"] = checkpoint
            _args["pretrain"]["write_tag"] = write_tag

            args_eval += [_args]

    return eval_nodes, eval_tasks_per_node, args_eval


def _interpolate_pos_embed(pos_embed_checkpoint, pos_embed_model, old_num_frames, new_num_frames, tubelet_size, patch_size, img_size):
    """
    Interpolate positional embeddings when num_frames changes between
    pre-training and fine-tuning.

    :param pos_embed_checkpoint: pos_embed tensor from checkpoint, shape (1, N_old, D)
    :param pos_embed_model: pos_embed tensor from current model, shape (1, N_new, D)
    :param old_num_frames: num_frames used during pre-training
    :param new_num_frames: num_frames used for fine-tuning
    :param tubelet_size: temporal patch size
    :param patch_size: spatial patch size
    :param img_size: spatial crop size (assumes square)
    """
    if pos_embed_checkpoint.shape == pos_embed_model.shape:
        return pos_embed_checkpoint

    dim = pos_embed_checkpoint.shape[-1]
    old_t = old_num_frames // tubelet_size
    new_t = new_num_frames // tubelet_size
    grid_h = img_size // patch_size
    grid_w = img_size // patch_size

    logger.info(
        f"Interpolating pos_embed: old ({old_t}, {grid_h}, {grid_w}) -> new ({new_t}, {grid_h}, {grid_w})"
    )

    # Reshape to (1, D, old_t, grid_h, grid_w) for trilinear interpolation
    pos_embed = pos_embed_checkpoint.reshape(1, old_t, grid_h, grid_w, dim).permute(0, 4, 1, 2, 3)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed,
        size=(new_t, grid_h, grid_w),
        mode="trilinear",
        align_corners=False,
    )
    pos_embed = pos_embed.permute(0, 2, 3, 4, 1).reshape(1, -1, dim)
    return pos_embed


def _adapt_mask_tokens_from_checkpoint(checkpoint_dict, model_state_dict, key_prefix="backbone."):
    """
    Adapt mask_tokens: copy matching ones from checkpoint, skip mismatched ones
    so model keeps its initialized values.
    Returns a new dict with adapted mask_tokens.
    """
    ckpt_mask_keys = sorted([k for k in checkpoint_dict if f"{key_prefix}mask_tokens" in k])
    model_mask_keys = sorted([k for k in model_state_dict if f"{key_prefix}mask_tokens" in k])

    num_ckpt = len(ckpt_mask_keys)
    num_model = len(model_mask_keys)

    if num_ckpt == num_model:
        return checkpoint_dict

    logger.info(
        f"Adapting mask_tokens: checkpoint has {num_ckpt}, model expects {num_model}"
    )

    adapted = dict(checkpoint_dict)

    # Remove all checkpoint mask_token keys
    for k in ckpt_mask_keys:
        del adapted[k]

    # Copy existing tokens to new keys (reuse pre-trained weights where possible)
    for i in range(min(num_ckpt, num_model)):
        adapted[model_mask_keys[i]] = checkpoint_dict[ckpt_mask_keys[i]]

    # Remaining model mask_tokens (i >= num_ckpt) are not in adapted,
    # so they will be left at model's initialized values when using strict=False

    return adapted


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    is_anneal=False,
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = 0
    if not is_anneal:
        epoch = checkpoint["epoch"]

    # -- Detect fpc/pos_embed mismatch between checkpoint and current model
    encoder_ckpt = checkpoint["encoder"]
    predictor_ckpt = checkpoint["predictor"]

    # Check encoder pos_embed
    enc_pos_key = "backbone.pos_embed"
    enc_needs_interpolation = (
        enc_pos_key in encoder_ckpt
        and enc_pos_key in encoder.state_dict()
        and encoder_ckpt[enc_pos_key].shape != encoder.state_dict()[enc_pos_key].shape
    )

    # Check predictor pos_embed
    pred_pos_key = "backbone.predictor_pos_embed"
    pred_needs_interpolation = (
        pred_pos_key in predictor_ckpt
        and pred_pos_key in predictor.state_dict()
        and predictor_ckpt[pred_pos_key].shape != predictor.state_dict()[pred_pos_key].shape
    )

    if enc_needs_interpolation or pred_needs_interpolation:
        logger.info("Detected pos_embed shape mismatch (fpc changed). Performing interpolation...")

        # Infer old/new num_frames from pos_embed shapes and model config
        enc_backbone = encoder.module.backbone if hasattr(encoder, "module") else encoder.backbone
        patch_size = enc_backbone.patch_size
        tubelet_size = enc_backbone.tubelet_size
        img_size = enc_backbone.img_height  # assumes square
        grid_h = img_size // patch_size

        if enc_needs_interpolation:
            old_N = encoder_ckpt[enc_pos_key].shape[1]
            old_t = old_N // (grid_h * grid_h)
            old_num_frames = old_t * tubelet_size

            new_num_frames = enc_backbone.num_frames

            encoder_ckpt[enc_pos_key] = _interpolate_pos_embed(
                encoder_ckpt[enc_pos_key],
                encoder.state_dict()[enc_pos_key],
                old_num_frames=old_num_frames,
                new_num_frames=new_num_frames,
                tubelet_size=tubelet_size,
                patch_size=patch_size,
                img_size=img_size,
            )

        if pred_needs_interpolation:
            pred_backbone = predictor.module.backbone if hasattr(predictor, "module") else predictor.backbone
            old_N = predictor_ckpt[pred_pos_key].shape[1]
            old_t = old_N // (grid_h * grid_h)
            old_num_frames = old_t * tubelet_size

            new_num_frames = pred_backbone.num_frames

            predictor_ckpt[pred_pos_key] = _interpolate_pos_embed(
                predictor_ckpt[pred_pos_key],
                predictor.state_dict()[pred_pos_key],
                old_num_frames=old_num_frames,
                new_num_frames=new_num_frames,
                tubelet_size=tubelet_size,
                patch_size=patch_size,
                img_size=img_size,
            )

    # -- Adapt mask_tokens if count mismatch
    predictor_ckpt = _adapt_mask_tokens_from_checkpoint(
        predictor_ckpt,
        predictor.state_dict(),
        key_prefix="backbone.",
    )

    # -- loading encoder
    msg = encoder.load_state_dict(encoder_ckpt, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    # -- loading predictor
    msg = predictor.load_state_dict(predictor_ckpt, strict=False)
    logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        target_ckpt = checkpoint["target_encoder"]
        # Apply same pos_embed interpolation for target_encoder
        if enc_pos_key in target_ckpt and enc_pos_key in target_encoder.state_dict():
            if target_ckpt[enc_pos_key].shape != target_encoder.state_dict()[enc_pos_key].shape:
                enc_backbone = encoder.module.backbone if hasattr(encoder, "module") else encoder.backbone
                patch_size = enc_backbone.patch_size
                tubelet_size = enc_backbone.tubelet_size
                img_size = enc_backbone.img_height
                grid_h = img_size // patch_size
                old_N = target_ckpt[enc_pos_key].shape[1]
                old_t = old_N // (grid_h * grid_h)
                old_num_frames = old_t * tubelet_size
                new_num_frames = enc_backbone.num_frames
                target_ckpt[enc_pos_key] = _interpolate_pos_embed(
                    target_ckpt[enc_pos_key],
                    target_encoder.state_dict()[enc_pos_key],
                    old_num_frames=old_num_frames,
                    new_num_frames=new_num_frames,
                    tubelet_size=tubelet_size,
                    patch_size=patch_size,
                    img_size=img_size,
                )
        msg = target_encoder.load_state_dict(target_ckpt, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    # -- loading optimizer (skip if shape mismatch detected, since param shapes changed)
    if enc_needs_interpolation or pred_needs_interpolation:
        logger.warning(
            "Skipping optimizer state loading due to pos_embed shape mismatch. "
            "Optimizer will be re-initialized."
        )
    else:
        opt.load_state_dict(checkpoint["opt"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.info(f"loaded optimizers from epoch {epoch}")

    logger.info(f"read-path: {r_path}")

    # If shapes changed, reset epoch to 0 (fresh training schedule)
    if enc_needs_interpolation or pred_needs_interpolation:
        logger.info("Resetting epoch to 0 due to model shape changes.")
        epoch = 0

    del checkpoint

    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    uniform_power=False,
    use_mask_tokens=False,
    num_mask_tokens=2,
    zero_init_mask_tokens=True,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )
    encoder = MultiSeqWrapper(encoder)
    predictor = vit_pred.__dict__["vit_predictor"](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    predictor = PredictorMultiSeqWrapper(predictor)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    is_anneal,
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    ipe_scale=1.25,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
):
    param_groups = [
        {"params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {"params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1))},
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if not is_anneal:
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    else:
        scheduler = LinearDecaySchedule(
            optimizer,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
