from tqdm import tqdm

import torch
import numpy as np
from .radar import Radar
from PIL import Image
import math

# Phase 2 domain radar tuning
from genesis.domain.radar_tuning import get_radar_domain_config

torch.set_default_device('cuda')


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return torch.tensor(value, dtype=torch.float32)


def _flatten_pointcloud(pointcloud):
    pointcloud = _as_tensor(pointcloud)
    if pointcloud.ndim == 3 and pointcloud.shape[-1] == 3:
        return pointcloud.reshape(-1, 3)
    if pointcloud.ndim == 2 and pointcloud.shape[1] == 3:
        return pointcloud
    if pointcloud.ndim == 2 and pointcloud.shape[0] == 3:
        return pointcloud.T
    raise ValueError(f"Unsupported pointcloud shape: {tuple(pointcloud.shape)}")

def calculate_environment_points(environment_pir):
    """
    environment_pir: (H, W, 3) torch tensor, assumed to be on the correct device (e.g., CUDA)
    Returns: (H*W, 3) point cloud tensor in camera space
    """
    H, W, _ = environment_pir.shape
    device = environment_pir.device

    distance = environment_pir[:, :, 0] * 5 + 5  # [H, W]

    fov_rad = math.radians(130)    # IWR6843AOP FOV에 맞게 수정
    fx = W / (2 * math.tan(fov_rad / 2))
    fy = fx
    cx = W / 2
    cy = H / 2

    j = torch.arange(0, H, device=device).view(-1, 1).expand(H, W)  # rows
    i = torch.arange(0, W, device=device).view(1, -1).expand(H, W)  # cols

    x = (i - cx) / fx  # [H, W]
    y = (j - cy) / fy
    z = torch.ones_like(x, device=device)

    xyz = torch.stack((x, y, z), dim=-1) * distance.unsqueeze(-1)  # [H, W, 3]
    points = xyz.reshape(-1, 3)  # [H*W, 3]
    return points

def create_interpolator(_frames, _pointclouds, environment_pir, frame_rate=30, remove_zeros=True,
                        body_model: str = None):
    """
    Phase 2 enhanced version:
    - Accepts optional body_model to apply domain-specific RCS scaling and micro-velocity jitter.
    - The velocity channel already computed in pathtracer is used as base;
      we add small high-frequency radial perturbations for breathing/tail/fur effects.
    """
    num_frames = len(_frames)
    total_time = num_frames / frame_rate
    frames = [_as_tensor(frame) for frame in _frames]
    pointclouds = [_flatten_pointcloud(pc) for pc in _pointclouds]

    # Load domain radar config once
    radar_cfg = None
    if body_model is not None:
        try:
            radar_cfg = get_radar_domain_config(body_model)
        except Exception:
            radar_cfg = None

    rcs_scale = radar_cfg.rcs_scale if radar_cfg else 1.0
    micro_amp = radar_cfg.micro_doppler_amp if radar_cfg else 0.0

    if environment_pir is not None:
        # redue the size of environment PIR to reduce the memory usages
        environment_pir = environment_pir.resize((64, 64), resample=Image.Resampling.BILINEAR)
        environment_pir = torch.tensor(np.array(environment_pir), dtype=torch.float32) / 255.0
        environment_points = calculate_environment_points(environment_pir)
        environment_intensity = environment_pir[:, :, 1].flatten()

    def _filter_frame(frame, pointcloud):
        flatten_pir = frame.reshape(-1, 3)
        depth = flatten_pir[:, 0]
        intensity = flatten_pir[:, 1] * rcs_scale
        mask = (depth > 0.1) & (intensity > 0.1)
        filtered_points = pointcloud[mask]
        if environment_pir is not None:
            return (
                torch.cat((environment_intensity, intensity[mask]), dim=0),
                torch.cat((environment_points, filtered_points), dim=0),
            )
        return intensity[mask], filtered_points

    def interpolator(time):
        if time < 0 or time > total_time:
            raise ValueError("Invalid time value")

        frame_index = int(time * frame_rate)
        if frame_index >= num_frames - 1:
            return _filter_frame(frames[-1], pointclouds[-1])

        t = (time * frame_rate) % 1  # fractional part of time
        frame1 = frames[frame_index].clone()
        frame2 = frames[frame_index + 1].clone()

        pointcloud1 = pointclouds[frame_index].clone()
        pointcloud2 = pointclouds[frame_index + 1].clone()

        zero_depth_frame1 = frame1[:, :, 0] == 0
        zero_depth_frame2 = frame2[:, :, 0] == 0

        zero_depth_frame1_flat = zero_depth_frame1.reshape(-1)
        zero_depth_frame2_flat = zero_depth_frame2.reshape(-1)

        frame1[zero_depth_frame1] = frame2[zero_depth_frame1]
        frame2[zero_depth_frame2] = frame1[zero_depth_frame2]

        pointcloud1[zero_depth_frame1_flat] = pointcloud2[zero_depth_frame1_flat]
        pointcloud2[zero_depth_frame2_flat] = pointcloud1[zero_depth_frame2_flat]

        interpolated_frame = frame1 * (1 - t) + frame2 * t
        interpolated_pointcloud = pointcloud1 * (1 - t) + pointcloud2 * t

        # Phase 2: Domain-specific micro velocity jitter (adds realistic micro-Doppler)
        if micro_amp > 1e-4 and interpolated_pointcloud.shape[0] > 0:
            # Add small sinusoidal radial perturbation along sensor view direction (approx -Z)
            phase = time * (2 * np.pi)
            jitter = micro_amp * 0.6 * torch.sin(torch.tensor(phase + 0.7, device=interpolated_pointcloud.device))
            # Simple: modulate Z coordinate slightly -> changes tof -> Doppler
            z_jitter = jitter * torch.randn(interpolated_pointcloud.shape[0], device=interpolated_pointcloud.device) * 0.4
            interpolated_pointcloud[:, 2] += z_jitter

        return _filter_frame(interpolated_frame, interpolated_pointcloud)

    return interpolator




def generate_signal_frames(body_pirs, body_auxs, envir_pir, radar_config, body_model: str = None):
    """
    Phase 2: Now accepts body_model so the interpolator can apply
    domain-specific RCS scaling and micro-Doppler velocity jitter.
    """
    interpolator = create_interpolator(body_pirs, body_auxs, envir_pir, frame_rate=30, body_model=body_model)
    total_motion_frames = len(body_pirs)

    radar = Radar(radar_config)

    total_radar_frame = int(total_motion_frames / 30 * radar.frame_per_second)
    frames = []
    desc = f"Generating radar frames [{body_model}]" if body_model else "Generating radar frames"
    for i in tqdm(range(total_radar_frame), desc=desc):
        frame_mimo = radar.frameMIMO(interpolator, i * 1.0 / radar.frame_per_second)
        frames.append(frame_mimo.cpu().numpy())
    frames = np.array(frames)
    return frames
