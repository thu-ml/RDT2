import torch
import torch.nn.functional as F
import numpy as np
import scipy.spatial.transform as st


def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat


def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot


def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose


def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot


def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))


def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))


def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose


def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]


def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv


def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose


def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot


def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    out = vec / norm
    return out


def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-1)
    return out


def rot6d_to_mat_torch(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    out = torch.stack((b1, b2, b3), dim=-1)
    return out


def geodesic_loss(R_pred, R_target, reduce=True, eps=1e-7, return_degrees=False):
    """
    Compute geodesic loss (rotation error) between predicted and target rotation matrices.

    Args:
        R_pred: (..., 3, 3) batch of predicted rotation matrices
        R_target: (..., 3, 3) batch of ground truth rotation matrices

    Returns:
        loss: (...,) batch of geodesic loss values
    """
    # Compute relative rotation R_err = R_pred^T @ R_target
    R_err = torch.matmul(
        torch.transpose(R_pred, -2, -1), R_target)

    # Compute trace of R_err
    trace = torch.diagonal(R_err, dim1=-2, dim2=-1).sum(-1)  # (B,)

    # Compute geodesic distance using arccos((trace - 1) / 2)
    # Clamp to avoid numerical issues: valid range for acos is [-1, 1]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0+eps, 1.0-eps)

    loss = torch.acos(cos_theta)  # (B,)

    # Rescale to [0, 180] degrees if required
    if return_degrees:
        loss = loss / torch.pi * 180  # Normalize to [0, 180]

    if reduce:
        loss = loss.mean()
    return loss


def mat_to_rot6d(mat):
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    out = np.concatenate((col0, col1), axis=-1)
    return out


def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10


def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out
