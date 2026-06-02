"""Datasets for RF-Genesis downstream radar validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # Keep NumPy-only imports usable outside the training env.
    torch = None

    class Dataset:
        pass


DOMAIN_TO_LABEL = {"smpl": 0, "smil": 1, "dog": 2, "cat": 3}


def _load_pointcloud_array(path: Path) -> np.ndarray:
    pc = np.load(path, allow_pickle=True)
    if isinstance(pc, np.ndarray) and pc.dtype == object:
        pc = pc.tolist()
    if isinstance(pc, list):
        pc = pc[0] if pc else np.zeros((0, 6), dtype=np.float32)
    if isinstance(pc, np.ndarray) and pc.ndim == 3:
        pc = pc[0]
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim == 1:
        pc = pc.reshape(1, -1)
    return pc


def _fit_point_count(pc: np.ndarray, max_points: int, channels: int = 6) -> np.ndarray:
    if pc.shape[1] < channels:
        pad = np.zeros((pc.shape[0], channels - pc.shape[1]), dtype=pc.dtype)
        pc = np.concatenate([pc, pad], axis=1)
    elif pc.shape[1] > channels:
        pc = pc[:, :channels]

    if pc.shape[0] > max_points:
        pc = pc[:max_points]
    elif pc.shape[0] < max_points:
        pad = np.zeros((max_points - pc.shape[0], channels), dtype=pc.dtype)
        pc = np.concatenate([pc, pad], axis=0)
    return pc.astype(np.float32, copy=False)


class SyntheticRadarDataset(Dataset):
    """Loads RF-Genesis output folders for simple point-cloud downstream tasks."""

    def __init__(
        self,
        output_root: Union[str, Path],
        scenario_json_files: Optional[Sequence[Union[str, Path]]] = None,
        max_points: int = 256,
        channels: int = 6,
    ):
        if torch is None:
            raise ModuleNotFoundError("SyntheticRadarDataset requires torch. Install/run inside the RF-Genesis training environment.")
        self.output_root = Path(output_root)
        self.scenario_json_files = list(scenario_json_files or [])
        self.max_points = max_points
        self.channels = channels
        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Path]:
        if not self.output_root.exists():
            return []
        samples = []
        for sample_dir in sorted(p for p in self.output_root.iterdir() if p.is_dir()):
            for filename in ("radarllm_6d.npy", "pointclouds.npy"):
                pc_file = sample_dir / filename
                if pc_file.exists():
                    samples.append(pc_file)
                    break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_file = self.samples[idx]
        sample_dir = pc_file.parent
        pc = _fit_point_count(_load_pointcloud_array(pc_file), self.max_points, self.channels)

        body_model = "smpl"
        obj_file = sample_dir / "obj_diff.npz"
        if obj_file.exists():
            obj = np.load(obj_file, allow_pickle=True)
            if "body_model" in obj:
                body_model = str(obj["body_model"])

        return {
            "pc": torch.from_numpy(pc),
            "domain_label": torch.tensor(DOMAIN_TO_LABEL.get(body_model, 0), dtype=torch.long),
            "meta": {"path": str(pc_file), "body_model": body_model},
        }


class RealRadarDataset(Dataset):
    """Placeholder-compatible interface for user-provided real mmWave data."""

    def __init__(self, root: Union[str, Path], max_points: int = 256, channels: int = 6):
        if torch is None:
            raise ModuleNotFoundError("RealRadarDataset requires torch. Install/run inside the RF-Genesis training environment.")
        self.root = Path(root)
        self.max_points = max_points
        self.channels = channels
        self.samples = sorted(self.root.glob("*/pcs_6d.npy"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pc_file = self.samples[idx]
        pc = _fit_point_count(_load_pointcloud_array(pc_file), self.max_points, self.channels)
        return {
            "pc": torch.from_numpy(pc),
            "domain_label": torch.tensor(-1, dtype=torch.long),
            "meta": {"path": str(pc_file), "body_model": "real"},
        }


def collate_radar(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    if torch is None:
        raise ModuleNotFoundError("collate_radar requires torch. Install/run inside the RF-Genesis training environment.")
    batch = list(batch)
    out: Dict[str, Any] = {}
    out["pc"] = torch.stack([item["pc"] for item in batch], dim=0)
    out["domain_label"] = torch.stack([item["domain_label"] for item in batch], dim=0)
    out["meta"] = [item.get("meta", {}) for item in batch]
    return out
