"""
Shared micro-motion injection utilities.

Provides the "diffusion-based refinement to hallucinate plausible micro-motions
(e.g., tail wags or infant fidgeting)" hook from the spec, implemented as
lightweight parametric generators (no heavy diffusion at inference time).

These are called from the DomainAdapter (Phase 2) or directly from object_diff
after retargeting, before the pose is written to obj_diff.npz or fed to the
pathtracer.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def get_micro_motion_fn(profile: str):
    """Return a function(t, body_model, amp) -> Dict[joint_name, axis_angle_delta]."""
    if profile == "tail_wag":
        return _tail_wag
    if profile == "breathing_fidget":
        return _breathing_fidget
    if profile == "ear_twitch":
        return _ear_twitch
    return _noop


def _noop(*args, **kwargs):
    return None


def _tail_wag(t: float, body_model: str = "dog", amp: float = 0.035) -> Dict:
    """Returns a small additive rotation delta for the tail joints."""
    phase = t * 6.2
    return {
        "tail_base": np.array([0.0, 0.0, amp * np.sin(phase)], dtype=np.float32),
        "tail_mid":  np.array([0.0, 0.0, amp * 1.4 * np.sin(phase + 0.4)], dtype=np.float32),
    }


def _breathing_fidget(t: float, body_model: str = "smil", amp: float = 0.008) -> Dict:
    """Chest expansion + small random limb twitches (infant or adult)."""
    breath = amp * np.sin(t * 1.8)
    fidget = 0.6 * amp * np.sin(t * 11.0 + 1.3)
    return {
        "chest": np.array([breath * 0.6, 0.0, fidget * 0.3], dtype=np.float32),
        "spine": np.array([breath * 0.3, 0.0, 0.0], dtype=np.float32),
    }


def _ear_twitch(t: float, body_model: str = "cat", amp: float = 0.25) -> Dict:
    """Quick ear flicks (mostly for cats)."""
    if body_model != "cat":
        return {}
    flick = amp * np.exp(-((t * 3.0) % 4.0 - 0.1)**2 / 0.03) * np.sin(t * 40)
    return {"left_ear": np.array([flick, 0.0, 0.0], dtype=np.float32)}


def inject_micro_motions(
    pose: np.ndarray,
    t: float,
    body_model: str,
    profile: str,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Apply a micro-motion profile directly to the pose array (in-place safe copy).

    This is the main entry point used by the pipeline for "hallucinating"
    the fine details that improve point-cloud sparsity patterns and micro-Doppler.
    """
    fn = get_micro_motion_fn(profile)
    deltas = fn(t, body_model=body_model)
    if not deltas:
        return pose

    out = pose.copy()
    # Joint index → pose array position (joint i → dims [i*3 .. i*3+2]).
    # SMAL-verified indices from smal_CVPR2017.pkl kintree (see quadruped.py).
    # SMPL/SMIL spine/chest indices follow the standard SMPL-24 ordering.
    # "left_ear" is intentionally absent: the 33-joint SMAL model has no dedicated
    # ear joints, so _ear_twitch deltas are silently dropped.
    joint_map = {
        "tail_base": 25,  # SMAL: first tail joint (branches from root/pelvis)
        "tail_mid":  28,  # SMAL: mid-tail (4th of 7 tail joints)
        "chest":      6,  # SMPL/SMIL: Spine2 / SMAL: thorax
        "spine":      3,  # SMPL/SMIL: Spine1 / SMAL: spine_3
    }

    for joint_name, delta in deltas.items():
        j = joint_map.get(joint_name)
        if j is not None and j * 3 + 2 < out.shape[0]:
            out[j * 3 : j * 3 + 3] += delta * strength
    return out
