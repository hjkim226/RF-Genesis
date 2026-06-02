import os
import pickle
import struct
import sys
import types
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import torch


torch.set_default_device('cuda')


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMPL_ROOT = REPO_ROOT / "models" / "smpl_models"
DEFAULT_SMIL_MODEL = DEFAULT_SMPL_ROOT / "smil_web.pkl"
DEFAULT_SMAL_ROOT = DEFAULT_SMPL_ROOT
SMAL_BODY_MODELS = {"cat": 0, "dog": 1}


def _install_chumpy_pickle_stubs():
    """Allow loading legacy SMPL/SMIL pickles without importing chumpy."""
    module_names = [
        "chumpy",
        "chumpy.ch",
        "chumpy.reordering",
        "chumpy.linalg",
        "chumpy.utils",
    ]

    class ChumpyStub:
        def __init__(self, *args, **kwargs):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["state"] = state

    for name in module_names:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    for name in module_names:
        module = sys.modules[name]
        for class_name in [
            "Ch",
            "ChLambda",
            "MatVecMult",
            "Select",
            "SelectScalar",
            "depends_on",
        ]:
            if not hasattr(module, class_name):
                setattr(module, class_name, ChumpyStub)

    sys.modules["chumpy"].ch = sys.modules["chumpy.ch"]


def _as_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "toarray"):
        return value.toarray()
    if hasattr(value, "r"):
        return np.asarray(value.r)
    if hasattr(value, "x"):
        return np.asarray(value.x)
    if hasattr(value, "a"):
        base = _as_numpy(value.a)
        idxs = getattr(value, "idxs", None)
        shape = getattr(value, "preferred_shape", None)
        if idxs is not None and shape is not None:
            return base.reshape(-1)[idxs].reshape(shape)
        return base
    return np.asarray(value)


def _load_pickle(filename):
    _install_chumpy_pickle_stubs()
    with open(filename, "rb") as fp:
        return pickle.load(fp, encoding="latin1")


def _resolve_smpl_model(gender, model_root):
    root = Path(model_root)
    filenames = {
        "male": "basicModel_m_lbs_10_207_0_v1.0.0.pkl",
        "female": "basicModel_f_lbs_10_207_0_v1.0.0.pkl",
    }
    return root / filenames.get(gender, filenames["male"])


def _resolve_smal_root():
    return Path(os.environ.get("SMAL_MODEL_ROOT", DEFAULT_SMAL_ROOT))


def _resolve_smal_model():
    smal_root = _resolve_smal_root()
    return Path(os.environ.get("SMAL_MODEL_PATH", smal_root / "smal_CVPR2017.pkl"))


def _resolve_smal_data():
    smal_root = _resolve_smal_root()
    return Path(os.environ.get("SMAL_DATA_PATH", smal_root / "smal_CVPR2017_data.pkl"))


def _load_smal_shape(body_model):
    with open(_resolve_smal_data(), "rb") as fp:
        data = pickle.load(fp, encoding="latin1")
    return np.asarray(data["cluster_means"][SMAL_BODY_MODELS[body_model]], dtype=np.float32)


def _write_ply(filename, vertices, faces):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    with open(filename, "wb") as fp:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {vertices.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {faces.shape[0]}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        )
        fp.write(header.encode("ascii"))
        vertices.tofile(fp)
        for face in faces:
            fp.write(struct.pack("<Biii", 3, int(face[0]), int(face[1]), int(face[2])))


def _smal_vertices_to_rf(vertices):
    return np.stack([vertices[:, 0], vertices[:, 2], vertices[:, 1]], axis=1)


def _ground_vertices(vertices):
    vertices = vertices.copy()
    vertices[:, 1] -= vertices[:, 1].min()
    return vertices


def _needs_ply_refresh(mesh_path):
    if not mesh_path.exists():
        return True
    with open(mesh_path, "rb") as fp:
        header = fp.read(128)
    return b"format binary_little_endian" not in header


def ensure_body_mesh_file(body_model="smpl", gender="male"):
    if body_model == "smpl" and gender == "female":
        return REPO_ROOT / "models" / "female.ply"
    if body_model in ("smpl", "smil"):
        return REPO_ROOT / "models" / "male.ply"
    if body_model not in SMAL_BODY_MODELS:
        raise ValueError(f"Unknown body_model '{body_model}'. Expected 'smpl', 'smil', 'dog', or 'cat'.")

    mesh_path = REPO_ROOT / "models" / f"smal_{body_model}.ply"
    if not _needs_ply_refresh(mesh_path):
        return mesh_path

    model = _load_pickle(_resolve_smal_model())
    v_template = _as_numpy(model["v_template"]).astype(np.float32)
    shapedirs = _as_numpy(model["shapedirs"]).astype(np.float32)
    faces = _as_numpy(model["f"]).astype(np.int64)
    betas = _load_smal_shape(body_model)
    vertices = v_template + np.einsum("l,vkl->vk", betas, shapedirs)
    vertices = _smal_vertices_to_rf(vertices)
    vertices = _ground_vertices(vertices)
    _write_ply(mesh_path, vertices, faces)
    return mesh_path


def _batch_rodrigues(axis_angles):
    batch_size = axis_angles.shape[0]
    device = axis_angles.device
    dtype = axis_angles.dtype

    angles = torch.norm(axis_angles + 1e-8, dim=1, keepdim=True)
    directions = axis_angles / angles
    directions = directions.unsqueeze(-1)

    cos = torch.cos(angles).view(batch_size, 1, 1)
    sin = torch.sin(angles).view(batch_size, 1, 1)

    rx, ry, rz = torch.split(directions, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
    skew = torch.cat(
        [
            zeros, -rz, ry,
            rz, zeros, -rx,
            -ry, rx, zeros,
        ],
        dim=1,
    ).view(batch_size, 3, 3)

    identity = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return identity + sin * skew + (1.0 - cos) * torch.bmm(skew, skew)


def _make_transform(rotation, translation):
    batch_size = rotation.shape[0]
    bottom = torch.zeros(
        batch_size, 1, 4, dtype=rotation.dtype, device=rotation.device
    )
    bottom[:, :, 3] = 1.0
    return torch.cat([torch.cat([rotation, translation], dim=2), bottom], dim=1)


class BodyModelLayer(torch.nn.Module):
    def __init__(self, model_path, center_idx=0, vertex_transform=None, ground_vertices=False):
        super().__init__()
        model = _load_pickle(model_path)

        v_template = _as_numpy(model["v_template"]).astype(np.float32)
        shapedirs = _as_numpy(model["shapedirs"]).astype(np.float32)
        posedirs = _as_numpy(model["posedirs"]).astype(np.float32)
        j_regressor = _as_numpy(model["J_regressor"]).astype(np.float32)
        weights = _as_numpy(model["weights"]).astype(np.float32)
        faces = _as_numpy(model["f"]).astype(np.int64)
        kintree_table = _as_numpy(model["kintree_table"]).astype(np.int64)

        id_to_col = {int(kintree_table[1, i]): i for i in range(kintree_table.shape[1])}
        parents = [-1]
        for i in range(1, kintree_table.shape[1]):
            parents.append(id_to_col[int(kintree_table[0, i])])

        self.center_idx = center_idx
        self.vertex_transform = vertex_transform
        self.ground_vertices = ground_vertices
        self.num_betas = shapedirs.shape[-1]
        self.num_joints = kintree_table.shape[1]
        self.register_buffer("v_template", torch.tensor(v_template))
        self.register_buffer("shapedirs", torch.tensor(shapedirs))
        self.register_buffer("posedirs", torch.tensor(posedirs.reshape(-1, posedirs.shape[-1])))
        self.register_buffer("J_regressor", torch.tensor(j_regressor))
        self.register_buffer("weights", torch.tensor(weights))
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))
        self.register_buffer("th_faces", torch.tensor(faces, dtype=torch.long))
        self.kintree_table = kintree_table

    def forward(self, th_pose, th_betas=None):
        batch_size = th_pose.shape[0]
        device = self.v_template.device
        dtype = self.v_template.dtype
        th_pose = th_pose.to(device=device, dtype=dtype)

        th_pose = th_pose.view(batch_size, self.num_joints, 3)
        if th_betas is None:
            th_betas = torch.zeros(batch_size, self.num_betas, dtype=dtype, device=device)
        else:
            th_betas = th_betas.to(device=device, dtype=dtype)
            if th_betas.shape[1] < self.num_betas:
                pad = torch.zeros(
                    batch_size,
                    self.num_betas - th_betas.shape[1],
                    dtype=dtype,
                    device=device,
                )
                th_betas = torch.cat([th_betas, pad], dim=1)
            elif th_betas.shape[1] > self.num_betas:
                th_betas = th_betas[:, :self.num_betas]

        v_shaped = self.v_template + torch.einsum("bl,vkl->bvk", th_betas, self.shapedirs)
        joints = torch.einsum("jv,bvk->bjk", self.J_regressor, v_shaped)

        rot_mats = _batch_rodrigues(th_pose.reshape(-1, 3)).view(
            batch_size, self.num_joints, 3, 3
        )
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch_size, -1)
        pose_offsets = torch.matmul(pose_feature, self.posedirs.T).view(
            batch_size, -1, 3
        )
        v_posed = v_shaped + pose_offsets

        transforms = []
        transforms.append(_make_transform(rot_mats[:, 0], joints[:, 0:1].transpose(1, 2)))
        for i in range(1, self.num_joints):
            parent = int(self.parents[i].item())
            rel_joint = joints[:, i:i + 1] - joints[:, parent:parent + 1]
            transform_i = _make_transform(rot_mats[:, i], rel_joint.transpose(1, 2))
            transforms.append(torch.matmul(transforms[parent], transform_i))
        transforms = torch.stack(transforms, dim=1)
        transforms_global = transforms

        joints_homo = torch.cat(
            [joints, torch.zeros(batch_size, self.num_joints, 1, dtype=dtype, device=device)],
            dim=2,
        ).unsqueeze(-1)
        init_bone = torch.matmul(transforms, joints_homo)
        init_bone = torch.nn.functional.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        transforms = transforms - init_bone

        t = torch.matmul(self.weights, transforms.reshape(batch_size, self.num_joints, 16))
        t = t.view(batch_size, -1, 4, 4)
        v_homo = torch.cat(
            [v_posed, torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device)],
            dim=2,
        ).unsqueeze(-1)
        verts = torch.matmul(t, v_homo)[:, :, :3, 0]
        jtr = transforms_global[:, :, :3, 3]

        if self.center_idx is not None:
            center = jtr[:, self.center_idx:self.center_idx + 1]
            verts = verts - center
            jtr = jtr - center

        if self.vertex_transform == "smal_to_rf":
            verts = torch.stack([verts[:, :, 0], verts[:, :, 2], verts[:, :, 1]], dim=2)
            jtr = torch.stack([jtr[:, :, 0], jtr[:, :, 2], jtr[:, :, 1]], dim=2)

        if self.ground_vertices:
            ground = verts[:, :, 1].amin(dim=1, keepdim=True)
            verts = verts - torch.stack(
                [
                    torch.zeros_like(ground),
                    ground,
                    torch.zeros_like(ground),
                ],
                dim=2,
            )
            jtr = jtr - torch.stack(
                [
                    torch.zeros_like(ground),
                    ground,
                    torch.zeros_like(ground),
                ],
                dim=2,
            )

        return verts, jtr


def get_smpl_layer(body_model="smpl", gender="male", device="cuda"):
    vertex_transform = None
    ground_vertices = False
    if body_model == "smil":
        model_root = Path(os.environ.get("SMPL_MODEL_ROOT", DEFAULT_SMPL_ROOT))
        model_path = Path(os.environ.get("SMIL_MODEL_PATH", model_root / "smil_web.pkl"))
    elif body_model == "smpl":
        model_root = Path(os.environ.get("SMPL_MODEL_ROOT", DEFAULT_SMPL_ROOT))
        model_path = _resolve_smpl_model(gender, model_root)
    elif body_model in SMAL_BODY_MODELS:
        model_path = _resolve_smal_model()
        vertex_transform = "smal_to_rf"
        ground_vertices = True
    else:
        raise ValueError(f"Unknown body_model '{body_model}'. Expected 'smpl', 'smil', 'dog', or 'cat'.")

    return BodyModelLayer(
        model_path,
        center_idx=0,
        vertex_transform=vertex_transform,
        ground_vertices=ground_vertices,
    ).to(device)


def apply_translation(vertices, translation_matrix):
    ones = torch.ones(vertices.shape[0], 1)
    homogeneous_vertices = torch.cat([vertices, ones], dim=1)
    transformed_vertices = torch.matmul(homogeneous_vertices, translation_matrix.T)
    transformed_vertices_3d = transformed_vertices[:, :3]
    return transformed_vertices_3d


def call_smpl_layer(pose_params, shape_params, body, need_face=False, transform=None):
    vertices, _ = body(pose_params, th_betas=shape_params)

    if len(vertices.shape) == 3:
        if vertices.shape[0] == 1:
            vertices = vertices[0]
        else:
            raise NotImplementedError("mesh batch is supported, yet")

    if transform is not None:
        vertices = apply_translation(vertices, transform)

    vertices_mi = mi.TensorXf(vertices.cpu().numpy())

    if not need_face:
        return vertices_mi

    faces_mi = mi.TensorXf(body.th_faces.cpu().numpy())
    return vertices_mi, faces_mi
