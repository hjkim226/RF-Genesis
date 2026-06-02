"""Domain-specific motion retargeting and micro-motion injection for SMAL/SMIL.

Implements the "quadruped-aware retargeting module" and "soft-tissue deformation layer"
from the SMAL/SMIL extensions spec using the hybrid strategy:
- Base motion from Kimodo-SMPLX (or MDM) + constraints
- Post-process retarget + gait injection (no new model training required)
- Simple parametric micro-motions (tail wag, breathing, fidget, ear twitch)

All modules are optional and gated by --body-model and the new
--no-micro-motions flag so existing human pipelines are untouched.
"""

from .quadruped import retarget_to_smal_quadruped
from .infant import retarget_to_smil_infant, add_soft_tissue_deformation
from .micro_motion import inject_micro_motions

__all__ = [
    "retarget_to_smal_quadruped",
    "retarget_to_smil_infant",
    "add_soft_tissue_deformation",
    "inject_micro_motions",
]
