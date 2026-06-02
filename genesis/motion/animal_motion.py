"""
Animal motion generation backends for RF-Genesis.

Backend priority order (auto mode):
  1. AnimalML3DBackend  — retrieval from OmniMotionGPT dataset, no model inference needed
  2. ProceduralBackend  — sinusoidal trot (always available, original behavior)

AnimalML3D setup (no model training required — just the repo with its data zips):
    git clone https://github.com/USRC-SEA/OmniMotionGPT  <anywhere>
    # The zip files in data/ are used directly. No weights to download.

The retrieval backend:
  - Reads animals_smal_joints.zip  → 1240 clips of (T, 35, 3) SMAL joint positions
  - Reads motion_captions.zip      → paired text descriptions
  - Keyword-matches the user prompt to the nearest clip (no heavy deps)
  - Converts (T, 35, 3) joint positions → (T, 99) SMAL axis-angle via IK
"""

from __future__ import annotations

import abc
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# SMAL skeleton topology
# ---------------------------------------------------------------------------
# Standard 33-joint SMAL kintree: SMAL_PARENT[j] = parent joint (-1 = root)
# Topology from smal_CVPR2017.pkl and genesis/retargeting/quadruped.py:
#   Spine   : 0→1→2→3→4→5→6
#   FL leg  : 6→7→8→9→10
#   FR leg  : 6→11→12→13→14
#   Neck/Head: 6→15→16→32
#   RL leg  : 0→17→18→19→20
#   RR leg  : 0→21→22→23→24
#   Tail    : 0→25→26→27→28→29→30→31
SMAL_PARENT = [
    -1, 0, 1, 2, 3, 4, 5,    # root + spine (0-6)
     6, 7, 8, 9,              # front-left leg (7-10)
     6, 11, 12, 13,           # front-right leg (11-14)
     6, 15,                   # neck, head (15-16)
     0, 17, 18, 19,           # rear-left leg (17-20)
     0, 21, 22, 23,           # rear-right leg (21-24)
     0, 25, 26, 27, 28, 29, 30,  # tail chain (25-31)
    16,                       # face (32)
]

BACKEND_NAMES = ("animalml3d", "procedural")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class QuadrupedMotionBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True if the backend can be used without additional setup."""

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        num_frames: int,
        body_model: str = "dog",
        fps: float = 30.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (pose (T,99) SMAL axis-angle, root_translation (T,3))."""


# ---------------------------------------------------------------------------
# AnimalML3D retrieval backend
# ---------------------------------------------------------------------------

class AnimalML3DBackend(QuadrupedMotionBackend):
    """
    Retrieval-based backend using the AnimalML3D dataset bundled with OmniMotionGPT.

    No model training or inference required — clips are retrieved from the dataset
    by keyword-matching the user prompt to the paired text captions.

    Data files used (inside the OmniMotionGPT repo's data/ folder):
      animals_smal_joints.zip  — 1240 clips, each (T, 35, 3) SMAL joint positions
      motion_captions.zip      — paired text descriptions for each clip

    The repo is detected automatically in these locations (checked in order):
      1. <rfgen-root>/../OmniMotionGPT/          (sibling directory)
      2. <rfgen-root>/ext/OmniMotionGPT/         (ext/ subdirectory)

    Dataset animals and their clip prefixes:
      dog  : doggieMN, huskydog
      cat  : catBG
      (other quadrupeds are also available as fallback)
    """

    name = "animalml3d"

    # Locations to search for the OmniMotionGPT repo (checked in order)
    _SEARCH_PATHS = [
        REPO_ROOT.parent / "OmniMotionGPT",
        REPO_ROOT / "ext" / "OmniMotionGPT",
    ]

    # body_model → preferred clip name prefixes (subset of the dataset)
    _ANIMAL_PREFIXES: Dict[str, List[str]] = {
        "dog": ["doggieMN", "huskydog"],
        "cat": ["catBG", "leopardSLM", "pumaRW"],   # cat-like; catBG has only 1 clip
    }

    # Keyword synonyms to widen prompt matching
    _SYNONYMS = {
        "run":    ["run", "running", "sprint", "gallop", "chase"],
        "walk":   ["walk", "walking", "stroll", "wander", "pace"],
        "trot":   ["trot", "trotting", "jog", "jogging"],
        "sit":    ["sit", "sitting", "crouch", "crouching"],
        "lie":    ["lie", "lying", "sleep", "sleeping", "rest"],
        "jump":   ["jump", "jumping", "leap", "leaping"],
        "stand":  ["stand", "standing", "idle", "wait"],
        "eat":    ["eat", "eating", "drink", "drinking"],
        "attack": ["attack", "aggress", "aggressive", "fight"],
    }

    # ---------------------------------------------------------------------------

    def _find_repo(self) -> Optional[Path]:
        for p in self._SEARCH_PATHS:
            if (p / "data" / "animals_smal_joints.zip").exists():
                return p
        return None

    def is_available(self) -> bool:
        return self._find_repo() is not None

    def generate(
        self, prompt: str, num_frames: int, body_model: str = "dog", fps: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        repo = self._find_repo()
        if repo is None:
            raise RuntimeError(
                "AnimalML3D data not found. Clone OmniMotionGPT next to this repo:\n"
                "  git clone https://github.com/USRC-SEA/OmniMotionGPT  "
                f"{REPO_ROOT.parent / 'OmniMotionGPT'}"
            )

        data_dir = repo / "data"
        index = self._build_index(data_dir)
        clip_name = self._retrieve(prompt, body_model, index)
        joints_3d = self._load_clip(data_dir, clip_name)    # (T, 35, 3)

        # The dataset uses 35 joints; standard SMAL has 33.
        # The extra 2 joints are additional face landmarks — drop them.
        joints_33 = joints_3d[:, :33, :]

        pose, root_translation = joints3d_to_smal_axisangle(joints_33, num_frames)
        return pose, root_translation

    # ---------------------------------------------------------------------------
    # Index building (cached on instance after first call)

    def _build_index(self, data_dir: Path) -> Dict[str, List[str]]:
        if hasattr(self, "_index_cache"):
            return self._index_cache  # type: ignore[return-value]

        index: Dict[str, List[str]] = {}
        with zipfile.ZipFile(data_dir / "motion_captions.zip") as zf:
            for entry in zf.namelist():
                if not entry.endswith("screenshots.txt"):
                    continue
                # entry format: motion_captions/<clip_name>/screenshots.txt
                parts = entry.split("/")
                if len(parts) < 2:
                    continue
                clip = parts[1]
                raw = zf.read(entry).decode("utf-8", errors="replace")
                captions = [l.strip() for l in raw.splitlines() if l.strip()]
                if clip and captions:
                    index[clip] = captions

        self._index_cache = index
        return index

    # ---------------------------------------------------------------------------
    # Retrieval

    def _retrieve(
        self, prompt: str, body_model: str, index: Dict[str, List[str]]
    ) -> str:
        prefixes = self._ANIMAL_PREFIXES.get(body_model, [])

        # Filter by animal type; fall back to all clips if no match
        candidates = {
            clip: caps
            for clip, caps in index.items()
            if any(clip.startswith(p) for p in prefixes)
        } if prefixes else {}

        if not candidates:
            log.warning(
                "[AnimalML3D] No '%s' clips found; searching all %d clips.",
                body_model, len(index),
            )
            candidates = index

        expanded = self._expand_prompt(prompt)
        best_clip, best_score = "", -1.0

        for clip, captions in candidates.items():
            score = self._score(expanded, clip, captions)
            if score > best_score:
                best_score, best_clip = score, clip

        # Final fallback: first candidate
        if not best_clip:
            best_clip = next(iter(candidates))

        log.info(
            "[AnimalML3D] prompt='%s' → clip='%s' (score=%.2f)",
            prompt, best_clip, best_score,
        )
        return best_clip

    def _expand_prompt(self, prompt: str) -> List[str]:
        """Expand prompt words with synonyms for better recall."""
        tokens = re.findall(r"[a-z]+", prompt.lower())
        expanded = set(tokens)
        for word in tokens:
            for synonyms in self._SYNONYMS.values():
                if word in synonyms:
                    expanded.update(synonyms)
        return list(expanded)

    def _score(self, prompt_words: List[str], clip_name: str, captions: List[str]) -> float:
        """
        Combined score:
          - keyword overlap with captions (weighted 0.7)
          - keyword overlap with clip filename tokens (weighted 0.3)
        """
        prompt_set = set(prompt_words)
        name_words = set(re.findall(r"[a-z]+", clip_name.lower()))

        # Caption score: max overlap across all captions
        cap_score = 0.0
        for caption in captions:
            cap_words = set(re.findall(r"[a-z]+", caption.lower()))
            cap_score = max(
                cap_score,
                len(prompt_set & cap_words) / max(len(prompt_set), 1),
            )

        # Filename score
        name_score = len(prompt_set & name_words) / max(len(prompt_set), 1)

        return 0.7 * cap_score + 0.3 * name_score

    # ---------------------------------------------------------------------------
    # Clip loading

    def _load_clip(self, data_dir: Path, clip_name: str) -> np.ndarray:
        with zipfile.ZipFile(data_dir / "animals_smal_joints.zip") as zf:
            path = f"animals_smal_joints/{clip_name}.npy"
            return np.load(io.BytesIO(zf.read(path)))


# ---------------------------------------------------------------------------
# Procedural backend (always available)
# ---------------------------------------------------------------------------

class ProceduralBackend(QuadrupedMotionBackend):
    """
    Sinusoidal trot gait — zero external dependencies, always available.
    This is the original RF-Genesis quadruped motion and the final fallback.
    """

    name = "procedural"

    def is_available(self) -> bool:
        return True

    def generate(self, prompt, num_frames, body_model="dog", fps=30.0):
        from genesis.retargeting.quadruped import retarget_to_smal_quadruped
        dummy_pose = np.zeros((num_frames, 72), dtype=np.float32)
        dummy_trans = np.zeros((num_frames, 3), dtype=np.float32)
        dummy_trans[:, 2] = np.linspace(0.0, 2.5, num_frames)
        return retarget_to_smal_quadruped(dummy_pose, dummy_trans, body_model=body_model)


# ---------------------------------------------------------------------------
# 3-D joint positions → SMAL axis-angle (lightweight IK)
# ---------------------------------------------------------------------------

def joints3d_to_smal_axisangle(
    joints_3d: np.ndarray,
    target_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 3-D joint positions (T, 33, 3) to SMAL axis-angle (T, 99).

    For each joint j, compute the rotation that maps the parent→j bone
    direction in frame 0 (used as rest pose) onto that direction in frame t.
    The rotation is stored as an axis-angle vector at dims [j*3 .. j*3+2].

    Parameters
    ----------
    joints_3d     : (T, J, 3) or (J, 3) array; J ≤ 33 SMAL joints
    target_frames : desired output length T (clips are looped or trimmed)

    Returns
    -------
    pose             : (T, 99) SMAL axis-angle
    root_translation : (T,  3) root joint world position
    """
    if joints_3d.ndim == 2:
        joints_3d = joints_3d[None]          # single frame → (1, J, 3)

    T, J, _ = joints_3d.shape
    num_j = min(J, 33)

    # Loop clip to reach target_frames
    if T < target_frames:
        reps = (target_frames + T - 1) // T
        joints_3d = np.tile(joints_3d, (reps, 1, 1))[:target_frames]
        T = target_frames

    pose_out = np.zeros((T, 99), dtype=np.float32)
    root_translation = joints_3d[:, 0, :].astype(np.float32)

    # Rest-pose bone unit vectors (frame 0)
    rest_dirs: dict[int, np.ndarray] = {}
    for j in range(1, num_j):
        p = SMAL_PARENT[j]
        if p < 0:
            continue
        d = joints_3d[0, j] - joints_3d[0, p]
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            rest_dirs[j] = d / n

    for t in range(T):
        for j in range(1, num_j):
            p = SMAL_PARENT[j]
            if p < 0 or j not in rest_dirs:
                continue
            cur = joints_3d[t, j] - joints_3d[t, p]
            cur_n = float(np.linalg.norm(cur))
            if cur_n < 1e-6:
                continue
            cur = cur / cur_n

            axis = np.cross(rest_dirs[j], cur)
            sin_a = float(np.linalg.norm(axis))
            cos_a = float(np.dot(rest_dirs[j], cur))
            if sin_a < 1e-8:
                continue
            angle = np.arctan2(sin_a, cos_a)
            pose_out[t, j * 3: j * 3 + 3] = (axis / sin_a) * angle

    return _pad_or_trim(pose_out, target_frames), _pad_or_trim(root_translation, target_frames)


def _pad_or_trim(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) == n:
        return arr
    if len(arr) > n:
        return arr[:n]
    pad = np.repeat(arr[-1:], n - len(arr), axis=0)
    return np.concatenate([arr, pad])


# ---------------------------------------------------------------------------
# Registry & public API
# ---------------------------------------------------------------------------

_REGISTRY: list[QuadrupedMotionBackend] = [
    AnimalML3DBackend(),
    ProceduralBackend(),
]


def list_available_backends() -> list[str]:
    """Return names of installed/usable backends in priority order."""
    return [b.name for b in _REGISTRY if b.is_available()]


def generate_quadruped_motion(
    prompt: str,
    num_frames: int,
    body_model: str = "dog",
    fps: float = 30.0,
    backend: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a SMAL quadruped motion sequence.

    Parameters
    ----------
    prompt      : text describing the desired motion (e.g. "a dog running")
    num_frames  : number of frames to generate
    body_model  : "dog" or "cat"
    fps         : target frame rate (used for timing in some backends)
    backend     : "auto" | "animalml3d" | "procedural"
                  "auto" tries installed backends in priority order.

    Returns
    -------
    pose             : (T, 99) SMAL axis-angle
    root_translation : (T,  3)
    """
    if backend == "auto":
        for b in _REGISTRY:
            if not b.is_available():
                continue
            try:
                log.info("[AnimalMotion] Using backend: %s", b.name)
                return b.generate(prompt, num_frames, body_model, fps)
            except Exception as exc:
                log.warning(
                    "[AnimalMotion] Backend '%s' failed: %s  → trying next.",
                    b.name, exc,
                )
        raise RuntimeError("All backends failed — this should not happen (procedural is always available).")

    for b in _REGISTRY:
        if b.name == backend:
            if not b.is_available():
                raise RuntimeError(
                    f"Backend '{backend}' is not available. "
                    f"Available: {list_available_backends()}"
                )
            return b.generate(prompt, num_frames, body_model, fps)

    raise ValueError(f"Unknown backend '{backend}'. Options: {BACKEND_NAMES}")
