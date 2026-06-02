"""
SMIL infant retargeting + soft-tissue deformation layer.

Follows the "SMIL (Infants)" section of the spec:
- Pediatric motion prior emphasis (supine/lying bias, limb asymmetries, jerky spontaneous moves)
- Soft-tissue deformation (jiggle / fat distribution) that influences mmWave scattering
  differently from rigid adult meshes.

Implementation is deliberately lightweight and parametric (user choice for v1):
- No learned fetal blend shapes at runtime yet (can be added behind a flag later).
- Simple mass-spring style jiggle on torso + proximal limbs.
- Supine bias + random micro-kicks / head turns for "spontaneous movement" realism.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# SMIL uses the same 24-joint / 72-dim layout as SMPL (root + 21 body + 2 hand pads).
# Standard SMPL-24 joint ordering (Loper et al., 2015):
#   0 Pelvis  1 L_Hip   2 R_Hip   3 Spine1  4 L_Knee  5 R_Knee
#   6 Spine2  7 L_Ankle 8 R_Ankle 9 Spine3 10 L_Foot 11 R_Foot
#  12 Neck   13 L_Collar 14 R_Collar 15 Head 16 L_Shoulder 17 R_Shoulder
#  18 L_Elbow 19 R_Elbow 20 L_Wrist 21 R_Wrist 22 L_Hand 23 R_Hand

SMIL_JOINTS = {
    "root": 0,
    "left_hip": 1,       "right_hip": 2,
    "spine": 3,
    "left_knee": 4,      "right_knee": 5,
    "chest": 6,
    "left_ankle": 7,     "right_ankle": 8,
    "neck": 9,
    "head": 15,
    "left_shoulder": 16, "left_elbow": 18, "left_wrist": 20,
    "right_shoulder": 17, "right_elbow": 19, "right_wrist": 21,
}


def retarget_to_smil_infant(
    pose: np.ndarray,
    root_translation: np.ndarray,
    supine_bias: float = 0.7,
    asymmetry: float = 0.4,
    fidget_rate: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bias a generic (often standing/walking) motion toward plausible infant behavior.

    - Rotates/suppresses upright poses toward supine/lying.
    - Adds limb asymmetry and small spontaneous kicks / arm waves.
    - Does NOT change the 72-dim pose layout (SMIL-compatible).

    Parameters
    ----------
    supine_bias : 0..1
        How strongly to pull the torso toward lying on back.
    asymmetry : 0..1
        Strength of left/right limb difference (common in young infants).
    fidget_rate : Hz
        Frequency of small spontaneous movements.
    """
    T = pose.shape[0]
    out = pose.copy().astype(np.float32)
    trans = root_translation.copy().astype(np.float32)

    t = np.arange(T, dtype=np.float32) / 30.0

    # 1. Supine bias: rotate root so the infant lies on its back
    #    (negative pitch around x in RF y-up convention)
    if supine_bias > 0.05:
        supine_angle = supine_bias * 1.35   # ~77° toward lying
        out[:, 0] = out[:, 0] * (1 - 0.6 * supine_bias)          # damp original roll
        out[:, 1] = out[:, 1] * (1 - 0.6 * supine_bias) + supine_angle  # pitch
        # Lower the root so the back is near the ground
        trans[:, 1] -= supine_bias * 0.35

    # 2. Spontaneous limb fidget / kicks (asymmetric)
    kick_l = 0.9 * np.sin(2 * np.pi * fidget_rate * 1.7 * t) * asymmetry
    kick_r = 0.9 * np.sin(2 * np.pi * fidget_rate * 2.1 * t + 1.3) * (1 - asymmetry) * 0.7

    # Legs (hips + knees) — very common infant movement when supine
    out[:, SMIL_JOINTS["left_hip"] * 3 + 1] += kick_l * 0.6
    out[:, SMIL_JOINTS["left_knee"] * 3 + 1] += kick_l * 1.1
    out[:, SMIL_JOINTS["right_hip"] * 3 + 1] += kick_r * 0.6
    out[:, SMIL_JOINTS["right_knee"] * 3 + 1] += kick_r * 1.1

    # Arms (random-ish waving)
    wave = 0.6 * np.sin(2 * np.pi * fidget_rate * 3.3 * t + 0.7)
    out[:, SMIL_JOINTS["left_elbow"] * 3 + 2] += wave * 0.4 * asymmetry
    out[:, SMIL_JOINTS["right_elbow"] * 3 + 2] += wave * 0.35 * (1 - asymmetry)

    # 3. Head turning (looking around while lying)
    head_turn = 0.35 * np.sin(2 * np.pi * 0.4 * t)
    out[:, SMIL_JOINTS["head"] * 3 + 2] += head_turn

    return out, trans


def add_soft_tissue_deformation(
    vertices: np.ndarray,          # (V, 3) or (T, V, 3) — usually per-frame from SMPL layer
    body_vel: Optional[np.ndarray] = None,  # (T, V, 3) or None — finite-diff velocity
    beta: Optional[np.ndarray] = None,      # (20,) SMIL shape — modulates amplitude
    strength: float = 0.7,
) -> np.ndarray:
    """
    Simple parametric soft-tissue jiggle.

    Adds low-frequency, velocity-dependent offsets to torso and proximal limbs.
    This changes the instantaneous surface the ray-tracer sees, producing
    more realistic micro-Doppler and point-cloud "fuzz" for infants vs rigid adults.

    The effect is deliberately cheap (no FEM) but directionally correct per the spec.
    """
    if vertices.ndim == 2:
        vertices = vertices[None, ...]   # (1, V, 3)
        squeeze = True
    else:
        squeeze = False

    T, V, _ = vertices.shape
    out = vertices.copy()

    if beta is None:
        beta = np.zeros(20, dtype=np.float32)

    # Amplitude scales with overall body size (first few SMIL betas roughly control mass)
    mass_scale = 0.6 + 0.8 * np.clip(beta[:3].mean(), -2.0, 2.0)

    # Very rough torso mask (works because SMIL v_template has similar vertex ordering to SMPL)
    # In production you would use a proper skinning weight or segmentation.
    torso_mask = np.zeros(V, dtype=bool)
    torso_mask[3000:6500] = True   # heuristic central band on the SMIL mesh

    if body_vel is None:
        # Fallback: finite difference on the vertex sequence
        body_vel = np.zeros_like(vertices)
        if T > 1:
            body_vel[1:] = np.diff(vertices, axis=0)
            body_vel[0] = body_vel[1]

    # Simple damped oscillator per-vertex (low-pass + inertia)
    jiggle = np.zeros_like(vertices)
    for t in range(T):
        v = body_vel[t]
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        # Jiggle direction = velocity perpendicular to surface normal (approx radial)
        radial = v / (speed + 1e-6)
        amp = strength * mass_scale * np.tanh(speed * 18.0) * 0.012   # meters
        jiggle[t] = radial * amp[:, :1] * (torso_mask[:, None].astype(np.float32))

    out = out + jiggle

    if squeeze:
        out = out[0]
    return out.astype(np.float32)
