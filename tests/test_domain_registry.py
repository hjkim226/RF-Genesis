"""Lightweight tests for the BodyDomain registry (Phase 0).

These tests require only numpy + stdlib and can run outside the full RF-Genesis
conda environment. They enforce the invariants needed for the SMAL/SMIL spec:
- Correct pose/shape dimensions per model
- Proper domain tags (quadruped / infant)
- SMAL cluster loading (when model files present)
- No regression in the four supported body models
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make RF-Genesis root importable when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from genesis.domain.registry import (
    BODY_DOMAINS,
    get_domain,
    get_pose_dim,
    get_shape_dim,
    default_shape_for,
    is_supported_body_model,
    SMAL_BODY_MODELS,
    load_smal_cluster_betas,
)


def test_all_four_models_registered():
    for name in ["smpl", "smil", "dog", "cat"]:
        assert is_supported_body_model(name), f"{name} missing from registry"
        d = get_domain(name)
        assert d.name == name


def test_pose_dimensions_match_spec_and_legacy():
    # From the SMAL/SMIL extension spec and existing RF-Genesis code
    assert get_pose_dim("smpl") == 72
    assert get_pose_dim("smil") == 72
    assert get_pose_dim("dog") == 99
    assert get_pose_dim("cat") == 99


def test_shape_dimensions():
    assert get_shape_dim("smpl") == 10
    assert get_shape_dim("smil") == 20
    # SMAL uses cluster means (dim typically 4-9 depending on data.pkl)
    for animal in SMAL_BODY_MODELS:
        d = get_domain(animal)
        assert d.shape_dim >= 4


def test_domain_tags():
    assert not get_domain("smpl").is_infant
    assert not get_domain("smpl").is_quadruped

    assert get_domain("smil").is_infant
    assert not get_domain("smil").is_quadruped

    assert get_domain("dog").is_quadruped
    assert get_domain("cat").is_quadruped


def test_default_shapes_are_correct_length_and_type():
    for name in ["smpl", "smil", "dog", "cat"]:
        shape = default_shape_for(name)
        assert isinstance(shape, np.ndarray)
        assert shape.dtype == np.float32
        assert shape.shape[0] == get_shape_dim(name)


def test_smal_cluster_betas_load_when_data_available():
    """Only runs if the SMAL data file is present (common in full setups)."""
    data_path = Path(
        os.environ.get(
            "SMAL_DATA_PATH",
            REPO_ROOT / "models" / "smpl_models" / "smal_CVPR2017_data.pkl",
        )
    )
    if not data_path.exists():
        pytest.skip("SMAL data file not present; skipping cluster load test")

    for animal in SMAL_BODY_MODELS:
        betas = load_smal_cluster_betas(animal)
        assert isinstance(betas, np.ndarray)
        assert betas.dtype == np.float32
        assert betas.ndim == 1
        # Should be non-zero (real cluster means)
        assert np.any(betas != 0)


def test_registry_is_single_source_no_hardcoded_duplicates():
    """Smoke that the legacy constants are now thin wrappers (no drift possible)."""
    # If these ever diverge, the unification in Phase 0 has been broken
    for name in BODY_DOMAINS:
        d = get_domain(name)
        assert get_pose_dim(name) == d.pose_dim
        assert get_shape_dim(name) == d.shape_dim
