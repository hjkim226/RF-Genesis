"""Domain adapters and registry for SMPL / SMIL / SMAL body models.

This package provides the modular extension points described in the
SMAL/SMIL extensions spec:
- BodyDomain registry (joint counts, shape spaces, domain-specific scaling)
- RadarDomainConfig for RCS, micro-Doppler, and material proxies
- Retargeting + micro-motion injection (Phase 1)
"""

from .registry import (
    BodyDomain,
    BODY_DOMAINS,
    get_domain,
    is_supported_body_model,
    SMPL_BODY_MODELS,
    SMAL_BODY_MODELS,
)
from .radar_tuning import (
    RadarDomainConfig,
    get_radar_domain_config,
    get_rcs_scale,
    get_micro_doppler_amp,
    get_reflectance_tint,
    get_point_density_scale,
)

__all__ = [
    "BodyDomain",
    "BODY_DOMAINS",
    "get_domain",
    "is_supported_body_model",
    "SMPL_BODY_MODELS",
    "SMAL_BODY_MODELS",
    "RadarDomainConfig",
    "get_radar_domain_config",
    "get_rcs_scale",
    "get_micro_doppler_amp",
    "get_reflectance_tint",
    "get_point_density_scale",
]
