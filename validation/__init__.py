"""Validation and metrics for synthetic radar data (SMPL/SMIL/SMAL).

Supports the cross-domain validation strategy described in the SMAL/SMIL
extensions spec:
- Point cloud fidelity (Chamfer, Hausdorff, density histograms)
- Micro-Doppler & dynamics (spectrogram SSIM, velocity distribution error)
- Downstream utility (synthetic → real transfer for HAR, pose estimation, etc.)
"""

from .metrics import (
    chamfer_distance,
    hausdorff_distance,
    point_density_histogram,
    doppler_histogram_error,
    spectrogram_ssim,
)

__all__ = [
    "chamfer_distance",
    "hausdorff_distance",
    "point_density_histogram",
    "doppler_histogram_error",
    "spectrogram_ssim",
]
