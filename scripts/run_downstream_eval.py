#!/usr/bin/env python3
"""
Main script for running downstream evaluation experiments on synthetic radar data.

Example usage (after generating some data with --body-model smpl/smil/dog/cat):

    python scripts/run_downstream_eval.py \
        --output-root RF-Genesis/output \
        --scenarios RF-Genesis/infant_scenarios.json RF-Genesis/pet.json \
        --task domain_classification \
        --epochs 20 \
        --batch-size 32 \
        --max-points 128

This script is intentionally simple and self-contained so it can be used
quickly for paper experiments.
"""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from validation.datasets import SyntheticRadarDataset, collate_radar
from validation.models.pointnet import PointNetClassifier
from validation.tasks import DomainClassificationTask
from validation.engine import train_one_epoch, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True,
                        help="Path to RF-Genesis/output directory containing scenario folders")
    parser.add_argument("--scenarios", type=str, nargs="+", default=[],
                        help="Paths to scenario JSON files (infant_scenarios.json, pet.json, etc.)")
    parser.add_argument("--task", type=str, default="domain_classification",
                        choices=["domain_classification", "pose_regression"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=== Downstream Evaluation Harness ===")
    print(f"Output root : {args.output_root}")
    print(f"Scenarios   : {args.scenarios}")
    print(f"Task        : {args.task}")
    print(f"Device      : {args.device}")

    # Dataset
    dataset = SyntheticRadarDataset(
        output_root=args.output_root,
        scenario_json_files=args.scenarios,
        max_points=args.max_points,
    )

    if len(dataset) == 0:
        print("No valid data found. Make sure you have run generations with --body-model smil/dog/cat etc.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_radar)

    # Model + Task
    if args.task == "domain_classification":
        model = PointNetClassifier(in_channels=6, num_classes=4).to(args.device)
        task = DomainClassificationTask(num_classes=4)
    else:
        raise NotImplementedError("pose_regression stub not fully wired in this quick harness yet")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nStarting training for {args.epochs} epochs on {len(dataset)} samples...\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, task, loader, optimizer, device=args.device)
        print(f"Epoch {epoch:02d} | Train loss: {train_metrics['loss']:.4f} | Acc: {train_metrics.get('accuracy', 0):.3f}")

    # Final eval on same data (for quick demo; use proper splits in real experiments)
    final_metrics = evaluate(model, task, loader, device=args.device)
    print("\n=== Final Metrics (train set for demo) ===")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nDone. In a real paper run you would use proper train/val/test splits and domain transfer setups.")


if __name__ == "__main__":
    main()
