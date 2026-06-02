"""
Task definitions for downstream radar point cloud learning.

Each task knows its loss, metrics, and how to interpret model outputs.
"""

from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import accuracy


class BaseTask:
    def __init__(self, name: str):
        self.name = name

    def compute_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def compute_metrics(self, pred: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


class DomainClassificationTask(BaseTask):
    """Task for classifying body domain (smpl / smil / dog / cat)."""

    def __init__(self, num_classes: int = 4):
        super().__init__("domain_classification")
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes

    def compute_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        target = batch["domain_label"]
        return self.criterion(pred, target)

    def compute_metrics(self, pred: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        target = batch["domain_label"]
        acc = accuracy(pred, target)
        return {"accuracy": acc}


class PoseRegressionTask(BaseTask):
    """Task for regressing pose + root translation."""

    def __init__(self):
        super().__init__("pose_regression")
        self.criterion = nn.MSELoss()

    def compute_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        target = batch["pose_target"]
        return self.criterion(pred, target)

    def compute_metrics(self, pred: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, float]:
        from .metrics import mpjpe
        # Very rough: treat last 3 dims as root, first 72 as pose angles (placeholder)
        # In real use you would do proper FK + joint error
        target = batch["pose_target"]
        mse = F.mse_loss(pred, target).item()
        return {"mse": mse}
