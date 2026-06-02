"""
Minimal training and evaluation engine for radar point cloud downstream tasks.

Designed to be readable and easy to extend for paper experiments.
"""

from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: torch.nn.Module,
    task,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda",
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    metrics_sum: Dict[str, float] = {}

    for batch in tqdm(dataloader, desc="Train", leave=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()
        pred = model(batch["pc"])
        loss = task.compute_loss(pred, batch)
        loss.backward()
        optimizer.step()

        bs = batch["pc"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        batch_metrics = task.compute_metrics(pred, batch)
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v * bs

    avg_loss = total_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    task,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics_sum: Dict[str, float] = {}

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        pred = model(batch["pc"])
        loss = task.compute_loss(pred, batch)

        bs = batch["pc"].size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        batch_metrics = task.compute_metrics(pred, batch)
        for k, v in batch_metrics.items():
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v * bs

    avg_loss = total_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in metrics_sum.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics
