"""
统计训练集中 delta_yaw 的分布直方图 (多进程加速版)。

用法:
    python stat_delta_yaw.py --data_path /disk/deepdata/dataset/vjepa_format_ad_data/train_front_split.txt

可选参数:
    --fps           视频采样帧率 (默认 4)
    --frameskip     帧间隔/tubelet_size (默认 2)
    --frames_per_clip  每clip帧数 (默认 8)
    --max_samples   最多统计多少条轨迹, -1 表示全部 (默认 -1)
    --num_workers   并行进程数 (默认 CPU 核心数)
    --output        输出图片路径 (默认 delta_yaw_distribution.png)
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def load_yaw_from_trajectory(path, fps, frameskip, frames_per_clip):
    """从单条轨迹的 h5 文件中提取 yaw 序列。

    Returns:
        yaw_sequence: [T_sampled] numpy array, 采样后的 yaw 值
        None: 如果加载失败
    """
    h5_path = os.path.join(path, "trajectory.h5")
    if not os.path.exists(h5_path):
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            cartesian_pos = f["observation/robot_state/cartesian_position"][:]
        yaw_all = cartesian_pos[:, 5]
        total_frames = len(yaw_all)

        nframes = frames_per_clip
        if total_frames < nframes:
            nframes = total_frames

        indices = np.arange(0, min(nframes, total_frames))
        indices = indices[:frames_per_clip]

        yaw_sampled = yaw_all[indices][::frameskip]
        return yaw_sampled

    except Exception:
        return None


def process_one_trajectory(path, fps, frameskip, frames_per_clip):
    """处理单条轨迹，返回 (clip_delta_yaw, pairwise_delta_yaws) 或 None。

    独立函数，供多进程 worker 调用。
    """
    yaw_seq = load_yaw_from_trajectory(path, fps, frameskip, frames_per_clip)
    if yaw_seq is None or len(yaw_seq) < 2:
        return None

    yaw_start = yaw_seq[0]
    yaw_end = yaw_seq[-1]
    clip_dy = float(np.arctan2(np.sin(yaw_end - yaw_start),
                               np.cos(yaw_end - yaw_start)))

    diffs = np.diff(yaw_seq)
    pw_dy = np.arctan2(np.sin(diffs), np.cos(diffs)).tolist()

    return clip_dy, pw_dy


def main():
    parser = argparse.ArgumentParser(description="统计训练集 delta_yaw 分布 (多进程)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="train.txt 路径, 每行一个轨迹目录")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=2, help="tubelet_size")
    parser.add_argument("--frames_per_clip", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="最多统计多少条, -1=全部")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="并行进程数, 0=自动 (CPU核心数)")
    parser.add_argument("--output", type=str, default="delta_yaw_distribution.png")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        samples = [line.strip() for line in f if line.strip()]

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"共 {len(samples)} 条轨迹, 使用 {num_workers} 个进程并行统计 ...")

    worker_fn = partial(
        process_one_trajectory,
        fps=args.fps,
        frameskip=args.frameskip,
        frames_per_clip=args.frames_per_clip,
    )

    clip_delta_yaws = []
    pairwise_delta_yaws = []
    failed = 0

    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, samples, chunksize=64),
            total=len(samples),
            desc="Loading trajectories",
        ):
            if result is None:
                failed += 1
                continue
            clip_dy, pw_dy = result
            clip_delta_yaws.append(clip_dy)
            pairwise_delta_yaws.extend(pw_dy)

    clip_delta_yaws = np.array(clip_delta_yaws)
    pairwise_delta_yaws = np.array(pairwise_delta_yaws)

    print(f"\n加载成功: {len(clip_delta_yaws)}, 失败: {failed}")
    print(f"相邻帧 delta_yaw 样本数: {len(pairwise_delta_yaws)}")

    # ── 统计摘要 ──
    deg = np.degrees(clip_delta_yaws)
    print(f"\n===== Clip-level delta_yaw (首尾差) =====")
    print(f"  mean  = {np.mean(deg):+.3f}°")
    print(f"  std   = {np.std(deg):.3f}°")
    print(f"  min   = {np.min(deg):+.3f}°")
    print(f"  max   = {np.max(deg):+.3f}°")
    print(f"  median= {np.median(deg):+.3f}°")

    # 按阈值分类统计
    STRAIGHT_THRESH = 0.15   # rad (~8.6°)
    UTURN_THRESH = 2.5       # rad (~143°)
    abs_dy = np.abs(clip_delta_yaws)
    n_straight = np.sum(abs_dy < STRAIGHT_THRESH)
    n_left = np.sum((clip_delta_yaws > STRAIGHT_THRESH) & (abs_dy < UTURN_THRESH))
    n_right = np.sum((clip_delta_yaws < -STRAIGHT_THRESH) & (abs_dy < UTURN_THRESH))
    n_uturn = np.sum(abs_dy >= UTURN_THRESH)

    total = len(clip_delta_yaws)
    print(f"\n===== 高层驾驶命令分布 (阈值: straight<{np.degrees(STRAIGHT_THRESH):.1f}°, uturn>{np.degrees(UTURN_THRESH):.1f}°) =====")
    print(f"  直行 (GO_STRAIGHT): {n_straight:>6d}  ({100*n_straight/total:.1f}%)")
    print(f"  左转 (TURN_LEFT):   {n_left:>6d}  ({100*n_left/total:.1f}%)")
    print(f"  右转 (TURN_RIGHT):  {n_right:>6d}  ({100*n_right/total:.1f}%)")
    print(f"  掉头 (U_TURN):      {n_uturn:>6d}  ({100*n_uturn/total:.1f}%)")

    # ── 绘图 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Delta Yaw Distribution  (N={total}, failed={failed})", fontsize=14)

    # 1) Clip-level delta_yaw 直方图 (弧度)
    ax = axes[0, 0]
    ax.hist(clip_delta_yaws, bins=200, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.axvline(STRAIGHT_THRESH, color="orange", linestyle="--", linewidth=0.8, label=f"±{STRAIGHT_THRESH:.2f} rad")
    ax.axvline(-STRAIGHT_THRESH, color="orange", linestyle="--", linewidth=0.8)
    ax.axvline(UTURN_THRESH, color="red", linestyle=":", linewidth=0.8, label=f"±{UTURN_THRESH:.1f} rad")
    ax.axvline(-UTURN_THRESH, color="red", linestyle=":", linewidth=0.8)
    ax.set_xlabel("delta_yaw (rad)")
    ax.set_ylabel("Count")
    ax.set_title("Clip-level delta_yaw (rad)")
    ax.legend()

    # 2) Clip-level delta_yaw 直方图 (角度)
    ax = axes[0, 1]
    ax.hist(deg, bins=200, color="coral", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    for thresh_deg in [np.degrees(STRAIGHT_THRESH), -np.degrees(STRAIGHT_THRESH)]:
        ax.axvline(thresh_deg, color="orange", linestyle="--", linewidth=0.8)
    ax.set_xlabel("delta_yaw (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Clip-level delta_yaw (degrees)")

    # 3) 相邻帧 delta_yaw (更精细粒度)
    ax = axes[1, 0]
    if len(pairwise_delta_yaws) > 0:
        ax.hist(pairwise_delta_yaws, bins=200, color="seagreen", edgecolor="none", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        pw_deg = np.degrees(pairwise_delta_yaws)
        ax.set_xlabel("pairwise delta_yaw (rad)")
        ax.set_ylabel("Count")
        ax.set_title(f"Frame-to-frame delta_yaw (N={len(pairwise_delta_yaws)})")
        stats_text = (f"mean={np.mean(pw_deg):+.4f}°\n"
                      f"std={np.std(pw_deg):.4f}°\n"
                      f"min={np.min(pw_deg):+.3f}°\n"
                      f"max={np.max(pw_deg):+.3f}°")
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 4) 饼图: 高层命令分布
    ax = axes[1, 1]
    labels = ["GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "U_TURN"]
    sizes = [n_straight, n_left, n_right, n_uturn]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if nonzero:
        ax.pie(
            [s for _, s, _ in nonzero],
            labels=[f"{l}\n{s} ({100*s/total:.1f}%)" for l, s, _ in nonzero],
            colors=[c for _, _, c in nonzero],
            autopct="",
            startangle=90,
        )
    ax.set_title("High-Level Driving Command Distribution")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存到: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
