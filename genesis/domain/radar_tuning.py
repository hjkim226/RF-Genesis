"""
Radar domain tuning profiles for SMPL / SMIL / SMAL.

Implements the "Radar Point Cloud Simulation Adaptations" section of the
SMAL/SMIL extensions spec:

- Domain-specific RCS scaling (animals produce stronger specular returns;
  infants produce weaker, more diffuse signals)
- Micro-Doppler modifiers (fur movement, breathing, tail wag)
- Material property proxies (permittivity / reflectance tint at ~60-77 GHz)
- Velocity field injection hooks for accurate Doppler

These are consumed by pathtracer.py (Mitsuba material/BSDF) and
signal_generator.py (intensity scaling + micro velocity modulation).

Single-radar only for v1 per user decision (multi-radar scaffolding left minimal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .registry import get_domain, BodyDomain


@dataclass(frozen=True)
class RadarDomainConfig:
    """Runtime radar simulation parameters for one body domain."""

    body_model: str
    rcs_scale: float
    micro_doppler_amp: float
    reflectance_tint: Tuple[float, float, float]

    # Fields with defaults must come after non-default fields
    breathing_freq_hz: float = 0.35
    tail_wag_freq_hz: float = 2.4
    point_density_scale: float = 1.0
    diffuse_noise_scale: float = 0.0


# Pre-computed profiles (can be overridden via env or JSON later)
_RADAR_PROFILES: Dict[str, RadarDomainConfig] = {}


def _build_default_profiles() -> Dict[str, RadarDomainConfig]:
    profiles = {}
    for name in ["smpl", "smil", "dog", "cat"]:
        domain = get_domain(name)

        # Base RCS from registry (already tuned in Phase 0)
        rcs = domain.rcs_multiplier

        # Extra domain-specific tweaks for radar physics
        if domain.is_quadruped:
            # Animals: stronger torso returns, fur causes some high-freq modulation
            rcs_scale = rcs * 1.1
            diffuse_noise = 0.03
            point_density = 0.95
        elif domain.is_infant:
            # Infants: much weaker, more diffuse, higher relative micro-motion
            rcs_scale = rcs * 0.85
            diffuse_noise = 0.18
            point_density = 0.75
        else:
            rcs_scale = rcs
            diffuse_noise = 0.02
            point_density = 1.0

        profiles[name] = RadarDomainConfig(
            body_model=name,
            rcs_scale=rcs_scale,
            micro_doppler_amp=domain.micro_doppler_amp,
            reflectance_tint=domain.material_reflectance_tint,
            point_density_scale=point_density,
            diffuse_noise_scale=diffuse_noise,
        )
    return profiles


def get_radar_domain_config(body_model: str) -> RadarDomainConfig:
    """Return (or lazily create) the radar tuning profile for a body model."""
    global _RADAR_PROFILES
    if not _RADAR_PROFILES:
        _RADAR_PROFILES = _build_default_profiles()

    if body_model not in _RADAR_PROFILES:
        # Fall back to smpl defaults but warn via the domain registry error
        from .registry import get_domain
        get_domain(body_model)  # will raise if truly unknown

    return _RADAR_PROFILES[body_model]


def get_rcs_scale(body_model: str) -> float:
    return get_radar_domain_config(body_model).rcs_scale


def get_micro_doppler_amp(body_model: str) -> float:
    return get_radar_domain_config(body_model).micro_doppler_amp


def get_reflectance_tint(body_model: str) -> Tuple[float, float, float]:
    return get_radar_domain_config(body_model).reflectance_tint


def get_point_density_scale(body_model: str) -> float:
    return get_radar_domain_config(body_model).point_density_scale
