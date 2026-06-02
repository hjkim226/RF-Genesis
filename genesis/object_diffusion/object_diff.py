import sys
import os
import subprocess
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import subprocess
from termcolor import colored
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from transforms3d.axangles import axangle2mat
import sys
sys.path.append("ext/mdm/")
from visualize.vis_utils import joints2smpl,npy2obj
from model.rotation2xyz import Rotation2xyz
import utils.rotation_conversions as geometry

# Domain registry (single source of truth for body model metadata)
from genesis.domain.registry import (
    BODY_DOMAINS,
    SMAL_BODY_MODELS,
    get_domain,
    get_pose_dim,
    default_shape_for,
    resolve_smal_data_path,
    load_smal_cluster_betas,
    get_micro_motion_profile,
)

# New domain retargeting + micro-motion (Phase 1)
from genesis.retargeting import (
    retarget_to_smal_quadruped,
    retarget_to_smil_infant,
    inject_micro_motions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------
# Thin wrappers around the domain registry (source of truth lives in
# genesis/domain/registry.py). Old names preserved for minimal diff.
# ------------------------------------------------------------------

def _resolve_smal_data_path():
    return resolve_smal_data_path()


def _smal_cluster_betas(body_model):
    return load_smal_cluster_betas(body_model)


def _pose_param_count_for(body_model):
    return get_pose_dim(body_model)


def _shape_params_for(body_model):
    # For SMAL this returns the cluster mean (not zeros). For others zeros.
    return default_shape_for(body_model)


def _retarget_pose_for(body_model, pose, num_frames):
    pose_param_count = _pose_param_count_for(body_model)
    pose = np.asarray(pose)
    if pose.ndim == 2 and pose.shape[1] == pose_param_count:
        return pose
    return np.zeros((num_frames, pose_param_count), dtype=np.float32)


def save_body_motion(out_dir, pose, root_translation, body_model="smpl", gender="male", skip_micro_motions=False):
    np.savez(
        out_dir + '/obj_diff.npz',
        pose=pose,
        shape=_shape_params_for(body_model),
        root_translation=root_translation,
        gender=gender,
        body_model=body_model,
        skip_micro_motions=skip_micro_motions,
    )


def _generate_quadruped(
    prompt, out_dir, body_model, animal_motion_backend, skip_micro_motions,
    num_frames=120, fps=30.0,
):
    """
    Bypass MDM entirely for quadruped (SMAL) models.
    Uses the animal motion backend to generate SMAL-native pose sequences,
    then optionally injects micro-motions before saving.
    """
    from termcolor import colored
    from genesis.motion.animal_motion import generate_quadruped_motion, list_available_backends

    available = list_available_backends()
    print(colored(
        f"[RFGen.AnimalMotion] Available backends: {available}  →  using '{animal_motion_backend}'",
        'cyan',
    ))

    pose, root_translation = generate_quadruped_motion(
        prompt=prompt,
        num_frames=num_frames,
        body_model=body_model,
        fps=fps,
        backend=animal_motion_backend,
    )

    if not skip_micro_motions:
        from genesis.retargeting.micro_motion import inject_micro_motions
        from genesis.domain.registry import get_micro_motion_profile
        profile = get_micro_motion_profile(body_model)
        if profile != "none":
            for i in range(len(pose)):
                pose[i] = inject_micro_motions(
                    pose[i], t=i / fps, body_model=body_model, profile=profile
                )

    save_body_motion(
        out_dir, pose, root_translation,
        body_model=body_model, gender="neutral",
        skip_micro_motions=skip_micro_motions,
    )


def retarget_body_model(out_dir, body_model="smpl", gender=None, skip_micro_motions=False):
    data = np.load(out_dir + '/obj_diff.npz', allow_pickle=True)
    gender = gender or (str(data['gender']) if 'gender' in data else 'male')
    root_translation = data['root_translation']
    pose = _retarget_pose_for(body_model, data['pose'], len(root_translation))
    save_body_motion(out_dir, pose, root_translation, body_model, gender, skip_micro_motions=skip_micro_motions)


def euler_to_axis_angle(euler_angles):
    """ Converts a set of Euler angles to axis-angle representation."""
    axis_angle_params = np.zeros_like(euler_angles)

    for i in range(euler_angles.shape[0]):
        for j in range(euler_angles.shape[1]):
            euler = euler_angles[i, j]
            r = R.from_euler('xyz', euler)
            axis_angle = r.as_rotvec()
            axis_angle_params[i, j] = axis_angle

    return axis_angle_params

def process(out_dir, body_model="smpl", gender="male", skip_micro_motions=False, animal_motion_backend="auto"):
    filename = out_dir+"/obj_diff_raw.npy"
    print(colored("---[RFGen.ObjDiff]:Runing SMPLify, it may take a few minutes.---", 'yellow'))
    print(colored("---[RFGen.ObjDiff]:This may be optimized in future updates.---", 'yellow'))
    data = np.load(filename,allow_pickle=True)
    motion = data[None][0]['motion'].transpose(0,3, 1, 2)
    
    num_frames = motion.shape[1]
    device='0'
    cuda=True
    
    os.chdir("ext/mdm")
    j2s = joints2smpl(num_frames=num_frames, device_id=device, cuda=cuda)
    os.chdir("../..")
    
    motion_tensor, opt_dict = j2s.joint2smpl(motion[0]) 
    thetas = motion_tensor[0, :-1, :, :num_frames]   
                                                # So basicly this would be the posture of SMPL, 
                                                # it is rot6d, but you can convert it to rotation matrix
                                                # see rotation2xyz
    root_translation = motion_tensor[0, -1, :3, :].cpu().numpy().transpose(1,0)


    thetas_matrix = thetas.transpose(2, 0).transpose(1, 2)
    thetas_matrix = geometry.rotation_6d_to_matrix(thetas_matrix)
    thetas_vec3 = geometry.matrix_to_euler_angles(thetas_matrix,"XYZ")
    thetas_vec3 = thetas_vec3.cpu().numpy()
    final_thetas = euler_to_axis_angle(thetas_vec3)
    smpl_params = final_thetas.reshape(final_thetas.shape[0], -1)
    if body_model in SMAL_BODY_MODELS:
        smpl_params = np.zeros((num_frames, _pose_param_count_for(body_model)), dtype=np.float32)
    
    # ------------------------------------------------------------------
    # Phase 1 hybrid retargeting (user-selected strategy)
    # If we have a non-human body_model and the user did not disable
    # micro-motions, run the domain-specific retarget + gait injection.
    # This turns the (mostly zero or human) pose into a plausible
    # quadruped trot or infant supine motion before saving.
    # ------------------------------------------------------------------
    if not skip_micro_motions:
        domain = get_domain(body_model)
        if domain.is_quadruped:
            # Quadruped path: this branch is only reached when process() is called
            # directly (e.g. retarget_body_model).  The primary generation path now
            # bypasses MDM and calls _generate_quadruped() instead.
            smpl_params, root_translation = retarget_to_smal_quadruped(
                smpl_params, root_translation, body_model=body_model
            )
        elif domain.is_infant:
            smpl_params, root_translation = retarget_to_smil_infant(
                smpl_params, root_translation
            )

        # Apply profile-driven micro-motions (tail wag, breathing, fidget)
        profile = get_micro_motion_profile(body_model)
        if profile != "none":
            for i in range(num_frames):
                smpl_params[i] = inject_micro_motions(
                    smpl_params[i], t=i / 30.0, body_model=body_model, profile=profile
                )

    save_body_motion(out_dir, smpl_params, root_translation, body_model, gender=gender,
                     skip_micro_motions=skip_micro_motions)
    


def generate(prompt, out_dir, body_model="smpl", gender="male", skip_micro_motions=False,
             animal_motion_backend="auto"):
    """
    Generate body motion for the given prompt and body model.

    For quadruped models (dog, cat) this bypasses MDM and calls the animal motion
    backend directly, avoiding the human-motion→SMAL retargeting mismatch.
    For all other models (smpl, smil) the original MDM pipeline is used.

    Parameters
    ----------
    animal_motion_backend : str
        One of "auto" | "animalml3d" | "procedural".
        Only used when body_model is a SMAL quadruped (dog / cat).
        "auto" tries installed backends in priority order, falls back to procedural.
    """
    if body_model in SMAL_BODY_MODELS:
        _generate_quadruped(
            prompt, out_dir,
            body_model=body_model,
            animal_motion_backend=animal_motion_backend,
            skip_micro_motions=skip_micro_motions,
        )
        return

    # Human / infant: use MDM as before
    os.chdir("ext/mdm/")
    subprocess.run(
        ['python', '-m', 'sample.generate_rfgen', '--model_path', './save/humanml_trans_enc_512/model000200000.pt',
         '--text_prompt', prompt,
         '--output_dir', "../../"+out_dir,
         '--num_samples', '1', '--num_repetitions', '1'])
    os.chdir("../..")
    process(out_dir, body_model=body_model, gender=gender, skip_micro_motions=skip_micro_motions)
    
