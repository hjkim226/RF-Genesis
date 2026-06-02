
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2
from tqdm import tqdm

from genesis.raytracing.radar import Radar 
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud, rangeFFT, dopplerFFT, process_pc, clutter_removal
from genesis.raytracing import smpl



# SMPL 
def display_smpl(
        model_info,
        model_faces=None,
        with_joints=False,
        kintree_table=None,
        ax=None,
        batch_idx=0,
        translation=None,
        title='SMPL model',
        ):
    """
    Displays mesh batch_idx in batch of model_info, model_info as returned by
    generate_random_model
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = model_info['verts'][batch_idx], model_info['joints'][
        batch_idx]
    if translation is not None:
        verts += translation
        joints += translation
    
    if model_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.2)
    else:
        mesh = Poly3DCollection(verts[model_faces], alpha=0.2)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if with_joints:
        draw_skeleton(joints, kintree_table=kintree_table, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    cx, cy, cz = (float(translation[0]), float(translation[1]), float(translation[2])) if translation is not None else (0.0, 0.0, 0.0)
    ax.set_xlim(cx - 2, cx + 2)
    ax.set_ylim(cy - 0.5, cy + 2)
    ax.set_zlim(cz - 1, cz + 3)
    ax.view_init(azim=-90, elev=100)
    ax.view_init(azim=30, elev=30, roll = 105)
    ax.set_title(title, fontsize=20)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return ax


def draw_skeleton(joints3D, kintree_table, ax=None, with_numbers=False):
    if ax is None:
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax

    left_right_mid = ['r', 'g', 'b']
    kintree_colors = [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1]
    for i in range(1, kintree_table.shape[1]):
        j1 = int(kintree_table[0][i])
        j2 = int(kintree_table[1][i])
        color_idx = kintree_colors[(i - 1) % len(kintree_colors)]
        ax.plot([joints3D[j1, 0], joints3D[j2, 0]],
                [joints3D[j1, 1], joints3D[j2, 1]],
                [joints3D[j1, 2], joints3D[j2, 2]],
                color=left_right_mid[color_idx], linestyle='-', linewidth=2, marker='o', markersize=5)
        if with_numbers:
            ax.text(joints3D[j2, 0], joints3D[j2, 1], joints3D[j2, 2], j2)
    return ax



def _display_translation(translation, body_model):
    if translation is None:
        return None
    translation = np.array(translation, dtype=np.float32).copy()
    if body_model in smpl.SMAL_BODY_MODELS:
        translation[1] = max(translation[1] - 1.0, 0.0)
    return translation


def draw_smpl_on_axis(pose, shape, translation=None, ax=None, body_model="smpl", gender="male"):
    pose = torch.tensor(pose).unsqueeze(0)
    shape = torch.tensor(shape).unsqueeze(0)
    smpl_layer = smpl.get_smpl_layer(body_model=body_model, gender=gender, device=pose.device)
    verts, Jtr = smpl_layer(pose, th_betas=shape)
    title = f"SMAL {body_model} model" if body_model in smpl.SMAL_BODY_MODELS else "SMPL model"
    translation = _display_translation(translation, body_model)

    display_smpl(
        {'verts': verts.cpu().detach(),
         'joints': Jtr.cpu().detach()},
        model_faces=smpl_layer.th_faces,
        with_joints=True,
        kintree_table=smpl_layer.kintree_table,translation = translation, ax = ax, title=title)
    

# Plotting Pointclouds
def draw_poinclouds_on_axis(pc,ax, tx,rx,elev,azim,title):
    pc = np.transpose(pc)
    ax.scatter(-pc[0], pc[1], pc[2], c=pc[4], cmap=plt.hot())
    if tx is not None:
        ax.scatter(tx[:,0], tx[:,2], tx[:,1], c="green", s= 50, marker =',', cmap=plt.hot())
    if rx is not None:
        ax.scatter(rx[:,0], rx[:,2], rx[:,1], c="orange", s= 50, marker =',', cmap=plt.hot())
    if pc.shape[1] > 0:
        y_max = max(float(pc[1].max()) * 1.2, 2.0)
    else:
        y_max = 6.0
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, y_max)
    ax.set_zlim(-0.5, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=20)

def draw_doppler_on_axis(radar_frame,pointcloud_cfg, ax):
    range_fft = rangeFFT(radar_frame,pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft,pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=20)

def draw_combined(i,pointcloud_cfg,radar_frames,pointclouds,smpl_data):
    smpl_frame_id = i               # 30FPS
    radar_frame_id = int(i/3)       # 10FPS

    poses = smpl_data["pose"]
    shape = smpl_data['shape']
    root_translation = smpl_data['root_translation']
    body_model = str(smpl_data['body_model']) if 'body_model' in smpl_data else 'smpl'
    gender = str(smpl_data['gender']) if 'gender' in smpl_data else 'male'


    fig= plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    draw_smpl_on_axis(poses[smpl_frame_id],shape,root_translation[smpl_frame_id],ax1,body_model,gender)


    ax2 = fig.add_subplot(132, projection='3d')
    draw_poinclouds_on_axis(pointclouds[radar_frame_id],ax2, None,None,30,-30,"Point clouds")


    ax3 = fig.add_subplot(133)
    draw_doppler_on_axis(radar_frames[radar_frame_id],pointcloud_cfg, ax3)


    plt.tight_layout()
    fig.canvas.draw()
    # 1. buffer_rgba()로 데이터를 가져옵니다. (RGBA 4개 채널)
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    
    # 2. reshape 시 마지막 숫자를 4로 설정합니다.
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
   
    # 3. RGBA에서 RGB만 남기기 위해 마지막 채널을 잘라냅니다.
    data = data[:, :, :3]
    
    plt.close(fig) 
    return data


def save_video(radar_cfg_file, radar_frames_file, smpl_data_file, output_file):
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    smpl_data = np.load(smpl_data_file,allow_pickle=True)

    # Process the pointclouds
    pointclouds = []
    radarllm_data = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        pointclouds.append(pc)
        # RadarLLM 포맷 변환: [x,y,z,i,v,r] -> [x,y,z,r,v,i]
        if pc.shape[0] > 0:
            pc_6d = pc[:, [0, 1, 2, 5, 4, 3]].copy()

            # intensity log scaling
            pc_6d[:, 5] = np.log1p(pc_6d[:, 5] * 1e10)
        else:
            pc_6d = np.empty((0, 6))

        radarllm_data.append(pc_6d)

    output_dir = os.path.dirname(output_file)

    os.makedirs(output_dir, exist_ok=True)

    # point cloud 저장
    np.save(os.path.join(output_dir, "pointclouds.npy"),
            np.array(pointclouds, dtype=object))

    np.save(os.path.join(output_dir, "radarllm_6d.npy"),
            np.array(radarllm_data, dtype=object))
    print(f"\nSaved {len(pointclouds)} frames to pointclouds.npy and radarllm_6d.npy")

    # --- NEW: also emit mmExpert format dense radar views (range/doppler/az vs slow-time) ---
    # This lets RF-Genesis outputs be used directly (or with minimal conversion) by mmExpert
    # for CLIP pretrain / LLM fine-tune. Uses the exact same range/doppler pipeline + virtual
    # array azimuth (padded to 128 bins to match mmExpert encoder expectations of 256/128/128).
    try:
        n_chirp = pointcloud_cfg.frameConfig.numLoopsPerFrame
        n_adc = pointcloud_cfg.frameConfig.numADCSamples
        range_time_list, dop_time_list, az_time_list = [], [], []
        for frame in radar_frames:  # each (3,4,128,256) or equiv
            r = rangeFFT(frame, n_adc)
            r = clutter_removal(r, axis=2)  # static clutter (as done for PC extraction)
            d = dopplerFFT(r, n_chirp)      # (3,4,dop,range)
            rp = np.abs(d).mean(axis=(0, 1, 2))   # range profile (256,)
            dp = np.abs(d).mean(axis=(0, 1, 3))   # doppler profile (128,)
            # azimuth profile via virtual array (12 elems) padded FFT -> 128 bins
            virt = d.reshape(12, d.shape[2], d.shape[3])
            apad = np.zeros((128, virt.shape[1], virt.shape[2]), dtype=complex)
            apad[:12, :, :] = virt
            afft = np.fft.fft(apad, axis=0)
            ap = np.abs(afft).mean(axis=(1, 2))
            ap = np.fft.fftshift(ap)
            range_time_list.append(rp)
            dop_time_list.append(dp)
            az_time_list.append(ap)
        range_t = np.stack(range_time_list, axis=0).T.astype(np.float32)  # (256, N)
        dop_t   = np.stack(dop_time_list,   axis=0).T.astype(np.float32)  # (128, N)
        az_t    = np.stack(az_time_list,    axis=0).T.astype(np.float32)  # (128, N)
        np.savez(os.path.join(output_dir, "mmexpert_views.npz"),
                 range_time=range_t, doppler_time=dop_t, azimuth_time=az_t)
        # Also save the conventional <name>.npz directly in the sample dir so it can be
        # consumed with minimal extra steps by mmExpert (filefolder=.../<name>, fileindex=<name>)
        sample_name = os.path.basename(output_dir)
        np.savez(os.path.join(output_dir, f"{sample_name}.npz"),
                 range_time=range_t, doppler_time=dop_t, azimuth_time=az_t)
        print(f"Saved mmExpert views to mmexpert_views.npz and {sample_name}.npz (shapes {range_t.shape})")
    except Exception as e:
        print(f"[mmExpert export] Skipped (could not compute views): {e}")
    
    # Write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = output_file
    out = cv2.VideoWriter(video_filename, fourcc, 30, (1200, 600))
    for i in tqdm(range(smpl_data["pose"].shape[0]-2)):
        frame = draw_combined(i,pointcloud_cfg,radar_frames,pointclouds,smpl_data)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    out.release()
