#!/usr/bin/env python3
"""
Quick before/after motion visualization script for paper figures.

Generates synthetic "human-like" base motion, then applies the new
domain retargeting + micro-motion modules (Phase 1), and produces
clean 3D comparison figures + optional GIFs.

No Mitsuba, no radar simulation, no heavy dependencies required.
Only needs: numpy + matplotlib.

Usage examples:
    python scripts/generate_motion_comparison_figures.py --body-model dog --out-dir paper_figures/dog_trot
    python scripts/generate_motion_comparison_figures.py --body-model smil --out-dir paper_figures/infant_fidget --gif

This directly supports the "Motion Generation and Parametric Model Integration"
and "Novelty in Pipeline Integration" sections of the SMAL/SMIL spec.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Import the new retargeting modules (pure numpy, very lightweight)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genesis.retargeting.quadruped import retarget_to_smal_quadruped
from genesis.retargeting.infant import retarget_to_smil_infant, add_soft_tissue_deformation
from genesis.retargeting.micro_motion import inject_micro_motions
from genesis.domain.registry import get_domain, get_micro_motion_profile


def generate_synthetic_human_motion(num_frames=90, fps=30):
    """
    Create a very simple "walking in place with arm swing" motion in SMPL 72-dim format.
    This is purely synthetic for figure generation — no external mocap or models needed.
    """
    t = np.arange(num_frames) / fps
    pose = np.zeros((num_frames, 72), dtype=np.float32)

    # Root forward lean + gentle bob
    pose[:, 1] = 0.12 * np.sin(2 * np.pi * 1.6 * t)          # pitch bob
    root_trans = np.zeros((num_frames, 3), dtype=np.float32)
    root_trans[:, 2] = 0.8 + 0.03 * np.sin(2 * np.pi * 1.6 * t)  # small vertical bob

    # Leg swing (hip + knee)
    for i, phase in enumerate([0.0, np.pi]):  # left / right
        hip_idx = 1 if i == 0 else 5
        knee_idx = 2 if i == 0 else 7
        pose[:, hip_idx * 3 + 1] = 0.55 * np.sin(2 * np.pi * 1.6 * t + phase)
        pose[:, knee_idx * 3 + 1] = 0.9 + 0.7 * np.sin(2 * np.pi * 1.6 * t + phase + np.pi)

    # Arm swing
    pose[:, 13 * 3 + 1] = 0.7 * np.sin(2 * np.pi * 1.6 * t)           # left shoulder
    pose[:, 16 * 3 + 1] = 0.7 * np.sin(2 * np.pi * 1.6 * t + np.pi)   # right shoulder

    # Subtle spine / head motion
    pose[:, 3 * 3 + 1] = 0.08 * np.sin(2 * np.pi * 1.6 * t)
    pose[:, 12 * 3 + 2] = 0.15 * np.sin(2 * np.pi * 0.9 * t)          # head turn

    return pose, root_trans


def get_skeleton_chains(body_model: str):
    """Return (parent, child) joint index pairs for simple stick figure drawing."""
    if body_model in ("smpl", "smil"):
        # Minimal but recognizable human skeleton (SMPL ordering)
        return [
            (0, 3), (3, 6), (6, 9), (9, 12),           # spine → head
            (9, 13), (13, 14), (14, 15),               # left arm
            (9, 16), (16, 17), (17, 18),               # right arm
            (0, 1), (1, 2), (2, 4),                    # left leg (hip-knee-ankle)
            (0, 5), (5, 7), (7, 8),                    # right leg
        ]
    else:
        # SMAL quadruped (using indices from retargeting/quadruped.py)
        return [
            (0, 1), (1, 2), (2, 3),                    # spine + head
            (0, 4), (4, 5), (5, 6),                    # tail
            (1, 7), (7, 8), (8, 9),                    # front left leg
            (1, 10), (10, 11), (11, 12),               # front right leg
            (0, 13), (13, 14), (14, 15),               # rear left leg
            (0, 16), (16, 17), (17, 18),               # rear right leg
        ]


def plot_skeleton(ax, pose, body_model="smpl", color="#1f77b4", alpha=0.9, linewidth=2.2):
    """
    Lightweight 3D stick figure using the same joint indices the retargeters expect.
    This version is good enough for clear before/after paper figures.
    """
    chains = get_skeleton_chains(body_model)
    n_joints = min(20, len(pose) // 3)

    # Build approximate joint positions from pose (axis-angle) — sufficient for visualization
    joints = np.zeros((n_joints, 3), dtype=np.float32)
    for j in range(1, n_joints):
        # Use pose angles to create plausible limb directions (very approximate FK)
        ax_ang = pose[j*3 : j*3+3]
        length = 0.18 if body_model in ("dog", "cat") else 0.22
        direction = np.array([
            0.4 * ax_ang[0],
            0.7 + 0.3 * ax_ang[1],
            0.3 * ax_ang[2]
        ])
        direction /= (np.linalg.norm(direction) + 1e-6)
        joints[j] = joints[j-1] + direction * length

    # Draw bones
    for parent, child in chains:
        if child < n_joints:
            ax.plot([joints[parent, 0], joints[child, 0]],
                    [joints[parent, 1], joints[child, 1]],
                    [joints[parent, 2], joints[child, 2]],
                    color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round")

    # Draw joints
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=18, alpha=alpha, edgecolors="white", linewidths=0.5)


def make_comparison_figure(base_pose, base_trans, retargeted_pose, retargeted_trans,
                           body_model, out_path, title_suffix=""):
    """Create a nice multi-panel before/after static figure."""
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    mid = len(base_pose) // 2

    # Before (human pose mapped naively onto target skeleton)
    plot_skeleton(ax1, base_pose[mid], body_model="smpl", color="#1f77b4")
    ax1.set_title(f"Input motion (human-like pose)\n{title_suffix}", fontsize=12)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.3, 1.8)
    ax1.set_zlim(-0.8, 1.2)
    ax1.view_init(elev=18, azim=-55)

    # After (retargeted + micro-motions)
    plot_skeleton(ax2, retargeted_pose[mid], body_model=body_model, color="#d62728", label="Retargeted + micro")
    ax2.set_title(f"After domain retargeting + micro-motions\n{body_model.upper()} — {title_suffix}", fontsize=12)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.3, 1.8)
    ax2.set_zlim(-0.8, 1.2)
    ax2.view_init(elev=18, azim=-55)

    fig.suptitle(f"SMAL/SMIL Domain Retargeting — {body_model.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved static comparison: {out_path}")


def make_animation(base_pose, base_trans, retargeted_pose, retargeted_trans,
                   body_model, out_path, fps=30, title=""):
    """Create an animated GIF comparing before and after over time."""
    fig = plt.figure(figsize=(12, 5.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    def update(frame):
        ax1.cla()
        ax2.cla()

        plot_skeleton(ax1, base_pose[frame], body_model="smpl", color="#1f77b4")
        ax1.set_title("Input (human-like)", fontsize=11)
        ax1.set_xlim(-1.3, 1.3)
        ax1.set_ylim(-0.4, 1.9)
        ax1.set_zlim(-0.9, 1.3)
        ax1.view_init(elev=16, azim=-50)

        plot_skeleton(ax2, retargeted_pose[frame], body_model=body_model, color="#d62728")
        ax2.set_title(f"Retargeted + micro-motions ({body_model})", fontsize=11)
        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.4, 1.9)
        ax2.set_zlim(-0.9, 1.3)
        ax2.view_init(elev=16, azim=-50)

        fig.suptitle(f"{title} — frame {frame}", fontsize=12)

    anim = FuncAnimation(fig, update, frames=len(base_pose), interval=1000/fps, blit=False)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close()
    print(f"Saved animation: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate before/after motion figures for SMAL/SMIL paper")
    parser.add_argument("--body-model", choices=["dog", "cat", "smil"], default="dog",
                        help="Target domain to demonstrate")
    parser.add_argument("--out-dir", type=str, default="paper_figures",
                        help="Output directory for figures")
    parser.add_argument("--frames", type=int, default=72, help="Number of frames to generate")
    parser.add_argument("--gif", action="store_true", help="Also generate animated GIF (slower)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain = get_domain(args.body_model)
    print(f"Generating comparison for {args.body_model} ({domain.display_name})")

    # 1. Synthetic base motion (human-like)
    base_pose, base_trans = generate_synthetic_human_motion(num_frames=args.frames)

    # 2. Apply the exact same retargeting used in the real pipeline
    if domain.is_quadruped:
        ret_pose, ret_trans = retarget_to_smal_quadruped(
            base_pose, base_trans, body_model=args.body_model
        )
        profile = get_micro_motion_profile(args.body_model)
        for i in range(len(ret_pose)):
            ret_pose[i] = inject_micro_motions(ret_pose[i], t=i/30.0,
                                               body_model=args.body_model, profile=profile)
    else:
        ret_pose, ret_trans = retarget_to_smil_infant(base_pose, base_trans)
        profile = get_micro_motion_profile(args.body_model)
        for i in range(len(ret_pose)):
            ret_pose[i] = inject_micro_motions(ret_pose[i], t=i/30.0,
                                               body_model=args.body_model, profile=profile)

    # 3. Generate figures
    suffix = "trotting_gait" if domain.is_quadruped else "supine_fidget"
    static_path = out_dir / f"{args.body_model}_comparison_{suffix}.png"
    make_comparison_figure(base_pose, base_trans, ret_pose, ret_trans,
                           args.body_model, static_path, title_suffix=suffix.replace("_", " "))

    if args.gif:
        gif_path = out_dir / f"{args.body_model}_comparison_{suffix}.gif"
        make_animation(base_pose, base_trans, ret_pose, ret_trans,
                       args.body_model, gif_path,
                       title=f"{domain.display_name} Retargeting")

    print("\nDone. Figures ready for paper.")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
