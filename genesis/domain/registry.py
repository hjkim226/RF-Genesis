"""Body domain registry for unified SMPL / SMIL / SMAL handling.

Consolidates scattered constants from object_diffusion/object_diff.py and
raytracing/smpl.py. Provides the single source of truth for:
- Pose / shape dimensionality per model
- SMAL family cluster lookup
- Domain tags (quadruped vs infant vs adult human)
- Extension hooks for Phase 1-2 (RCS scaling, micro-motion profiles, etc.)

This enables the "modular adapter" contribution in the SMAL/SMIL spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyDomain:
    """Metadata and parameters for one supported body model family."""

    name: str                    # canonical key: "smpl", "smil", "dog", "cat"
    display_name: str
    pose_dim: int                # axis-angle params (72 for SMPL/SMIL, 99 for SMAL)
    shape_dim: int               # beta count (10 SMPL, 20 SMIL, 41 for bundled SMAL clusters)
    is_quadruped: bool = False
    is_infant: bool = False
    default_gender: str = "male"

    # Radar / simulation tuning (Phase 2 hooks; defaults are neutral)
    rcs_multiplier: float = 1.0          # >1.0 for larger/stronger specular (animals)
    micro_doppler_amp: float = 0.0       # amplitude of injected micro radial vel (m/s)
    material_reflectance_tint: Tuple[float, float, float] = (0.8, 0.8, 0.8)  # Mitsuba BSDF proxy
    ray_density_scale: float = 1.0       # future: point subsampling multiplier

    # Motion / retargeting hooks (Phase 1)
    gait_type: str = "biped"             # "biped", "quadruped_trot", "infant_supine"
    micro_motion_profile: str = "none"   # "none", "tail_wag", "breathing_fidget", "ear_twitch"

    # Optional per-domain rest pose or prior path (populated lazily)
    rest_pose_path: Optional[str] = None
    pose_prior_path: Optional[str] = None

    # Canonical scale (meters). Used for normalization / CoM adjustment.
    canonical_height_m: float = 1.7


# ---------------------------------------------------------------------------
# Registry data
# ---------------------------------------------------------------------------

SMPL_BODY_MODELS = ("smpl", "smil")
SMAL_BODY_MODELS = ("dog", "cat")  # keys map to cluster indices in smal_CVPR2017_data.pkl

# SMAL family cluster indices (from smal_CVPR2017_data.pkl "cluster_means")
SMAL_CLUSTER_INDEX: Dict[str, int] = {"cat": 0, "dog": 1}

BODY_DOMAINS: Dict[str, BodyDomain] = {
    "smpl": BodyDomain(
        name="smpl",
        display_name="Adult Human (SMPL)",
        pose_dim=72,
        shape_dim=10,
        is_quadruped=False,
        is_infant=False,
        default_gender="male",
        rcs_multiplier=1.0,
        micro_doppler_amp=0.02,           # subtle breathing
        material_reflectance_tint=(0.82, 0.82, 0.82),
        gait_type="biped",
        micro_motion_profile="breathing_fidget",
        canonical_height_m=1.70,
    ),
    "smil": BodyDomain(
        name="smil",
        display_name="Infant (SMIL)",
        pose_dim=72,
        shape_dim=20,
        is_quadruped=False,
        is_infant=True,
        default_gender="neutral",
        rcs_multiplier=0.55,              # smaller body → weaker, more diffuse returns
        micro_doppler_amp=0.08,           # higher relative micro-motion (jerky limbs, breathing)
        material_reflectance_tint=(0.90, 0.88, 0.85),  # softer infant skin/clothing
        gait_type="infant_supine",
        micro_motion_profile="breathing_fidget",
        canonical_height_m=0.55,          # ~55 cm typical infant length
    ),
    "dog": BodyDomain(
        name="dog",
        display_name="Dog (SMAL canidae)",
        pose_dim=99,
        shape_dim=41,
        is_quadruped=True,
        is_infant=False,
        default_gender="neutral",
        rcs_multiplier=1.35,              # larger torso, lower height → stronger specular
        micro_doppler_amp=0.12,           # tail + fur + gait
        material_reflectance_tint=(0.65, 0.60, 0.55),  # fur darker / more absorbent at 60 GHz
        gait_type="quadruped_trot",
        micro_motion_profile="tail_wag",
        canonical_height_m=0.45,          # shoulder height proxy
    ),
    "cat": BodyDomain(
        name="cat",
        display_name="Cat (SMAL felidae)",
        pose_dim=99,
        shape_dim=41,
        is_quadruped=True,
        is_infant=False,
        default_gender="neutral",
        rcs_multiplier=1.15,
        micro_doppler_amp=0.10,
        material_reflectance_tint=(0.70, 0.68, 0.65),
        gait_type="quadruped_trot",
        micro_motion_profile="tail_wag",
        canonical_height_m=0.30,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_supported_body_model(body_model: str) -> bool:
    return body_model in BODY_DOMAINS


def get_domain(body_model: str) -> BodyDomain:
    """Return the BodyDomain record for a supported model.

    Raises ValueError for unknown models (consistent with existing checks).
    """
    if body_model not in BODY_DOMAINS:
        raise ValueError(
            f"Unknown body_model '{body_model}'. "
            f"Expected one of: {list(BODY_DOMAINS.keys())}"
        )
    return BODY_DOMAINS[body_model]


def get_pose_dim(body_model: str) -> int:
    return get_domain(body_model).pose_dim


def get_shape_dim(body_model: str) -> int:
    return get_domain(body_model).shape_dim


def get_smal_cluster_index(body_model: str) -> int:
    if body_model not in SMAL_BODY_MODELS:
        raise ValueError(f"'{body_model}' is not a SMAL model")
    return SMAL_CLUSTER_INDEX[body_model]


def resolve_smal_data_path() -> Path:
    """Centralized resolver (previously duplicated in object_diff.py and smpl.py)."""
    # Default matches existing RF-Genesis layout
    default_root = Path(__file__).resolve().parents[2] / "models" / "smpl_models"
    root = Path(os.environ.get("SMAL_MODEL_ROOT", default_root))
    return Path(os.environ.get("SMAL_DATA_PATH", root / "smal_CVPR2017_data.pkl"))


def load_smal_cluster_betas(body_model: str) -> np.ndarray:
    """Load the SMAL cluster mean beta vector for the given family (dog/cat)."""
    if body_model not in SMAL_BODY_MODELS:
        raise ValueError(f"load_smal_cluster_betas only valid for {SMAL_BODY_MODELS}")
    data_path = resolve_smal_data_path()
    import pickle
    with open(data_path, "rb") as fp:
        data = pickle.load(fp, encoding="latin1")
    idx = get_smal_cluster_index(body_model)
    return np.asarray(data["cluster_means"][idx], dtype=np.float32)


def default_shape_for(body_model: str) -> np.ndarray:
    """Return the canonical default shape vector (zeros or SMAL cluster mean)."""
    domain = get_domain(body_model)
    if body_model in SMAL_BODY_MODELS:
        return load_smal_cluster_betas(body_model)
    # SMPL: 10 zeros; SMIL: 20 zeros (matching historical behavior)
    return np.zeros(domain.shape_dim, dtype=np.float32)


# ---------------------------------------------------------------------------
# Small convenience for future micro-motion / retarget modules
# ---------------------------------------------------------------------------

def get_micro_motion_profile(body_model: str) -> str:
    return get_domain(body_model).micro_motion_profile


def get_gait_type(body_model: str) -> str:
    return get_domain(body_model).gait_type
