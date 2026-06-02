"""
Core quantitative metrics for synthetic vs real (or synthetic vs synthetic)
radar point clouds and micro-Doppler signatures.

Designed to support the claims in the SMAL/SMIL paper section:
- Point Cloud Fidelity: Chamfer, Hausdorff, density/Doppler histogram matching
- Micro-Doppler & Dynamics: Spectrogram SSIM, velocity distribution error
- Target: <15 cm error (as in original RF-Genesis baselines)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d


def chamfer_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """
    Symmetric Chamfer Distance between two point clouds (in meters).
    pc1, pc2: (N, 3) or (N, D) arrays.
    Lower is better. Typical good synthetic-real match: <0.15 m.
    """
    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)
    if pc1.size == 0 or pc2.size == 0:
        return float("inf")

    d12 = cdist(pc1, pc2).min(axis=1).mean()
    d21 = cdist(pc2, pc1).min(axis=1).mean()
    return float((d12 + d21) / 2.0)


def hausdorff_distance(pc1: np.ndarray, pc2: np.ndarray) -> float:
    """Directed + reverse Hausdorff (max min distance)."""
    pc1 = np.asarray(pc1, dtype=np.float64)
    pc2 = np.asarray(pc2, dtype=np.float64)
    if pc1.size == 0 or pc2.size == 0:
        return float("inf")

    d12 = cdist(pc1, pc2).min(axis=1).max()
    d21 = cdist(pc2, pc1).min(axis=1).max()
    return float(max(d12, d21))


def point_density_histogram(pc: np.ndarray, bins: int = 64, range_m: Tuple[float, float] = (0.0, 4.0)) -> np.ndarray:
    """1D histogram of radial distances (proxy for point density vs range)."""
    pc = np.asarray(pc)
    if pc.ndim == 2 and pc.shape[1] >= 3:
        dist = np.linalg.norm(pc[:, :3], axis=1)
    else:
        dist = pc.ravel()
    hist, _ = np.histogram(dist, bins=bins, range=range_m, density=True)
    return hist.astype(np.float32)


def doppler_histogram_error(pc1: np.ndarray, pc2: np.ndarray, bins: int = 64) -> float:
    """
    Earth-mover / L1 distance between velocity (Doppler) histograms.
    Assumes 4th column is velocity in m/s.
    """
    v1 = pc1[:, 3] if pc1.ndim == 2 and pc1.shape[1] > 3 else np.zeros(len(pc1))
    v2 = pc2[:, 3] if pc2.ndim == 2 and pc2.shape[1] > 3 else np.zeros(len(pc2))

    h1, edges = np.histogram(v1, bins=bins, density=True)
    h2, _ = np.histogram(v2, bins=edges, density=True)

    # L1 distance (simple, robust proxy for EMD on 1D)
    return float(np.abs(h1 - h2).sum() * (edges[1] - edges[0]))


def spectrogram_ssim(spec1: np.ndarray, spec2: np.ndarray, sigma: float = 1.5) -> float:
    """
    Simple structural similarity on (log) micro-Doppler spectrograms.
    Expects 2D arrays of shape (time, freq) or (doppler, time).
    Returns value in [-1, 1]; higher is better (1.0 = identical).
    """
    s1 = np.log1p(np.abs(spec1).astype(np.float64))
    s2 = np.log1p(np.abs(spec2).astype(np.float64))

    # Light Gaussian smoothing for robustness
    s1 = gaussian_filter1d(gaussian_filter1d(s1, sigma, axis=0), sigma, axis=1)
    s2 = gaussian_filter1d(gaussian_filter1d(s2, sigma, axis=0), sigma, axis=1)

    # Normalized cross-correlation style SSIM (simplified)
    mu1, mu2 = s1.mean(), s2.mean()
    var1, var2 = s1.var(), s2.var()
    cov = ((s1 - mu1) * (s2 - mu2)).mean()

    c1 = 1e-4
    c2 = 1e-3
    ssim = (2 * mu1 * mu2 + c1) * (2 * cov + c2) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    return float(np.clip(ssim, -1.0, 1.0))


# ------------------------------------------------------------------
# Convenience batch evaluator
# ------------------------------------------------------------------
def evaluate_pointcloud_pair(real_pc: np.ndarray, synth_pc: np.ndarray) -> dict:
    """Return a small dict of metrics useful for tables in the paper."""
    return {
        "chamfer_m": chamfer_distance(real_pc, synth_pc),
        "hausdorff_m": hausdorff_distance(real_pc, synth_pc),
        "doppler_hist_l1": doppler_histogram_error(real_pc, synth_pc),
    }
