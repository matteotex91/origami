import numpy as np
import numpy.linalg as la


def ext_mat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_mat(u, phi):
    return (
        np.tensordot(u, u, axes=0)
        + np.cos(phi) * (np.eye(3) - np.tensordot(u, u, axes=0))
        + np.sin(phi) * ext_mat(u)
    )


def comp_rot_mat(u_v, phi_v):
    if len(phi_v) == 1:
        return rot_mat(u_v[0], phi_v[0])
    else:
        return rot_mat(np.dot(comp_rot_mat(u_v[1:], phi_v[1:]), u_v[0]), phi_v[0])


def angle_mat(u_v, phi_v):
    R = np.eye(3)
    for i in range(len(phi_v)):
        R = np.dot(R, comp_rot_mat(u_v[i:], phi_v[i:]))
    return R


u_v = [
    np.array([1, 1, 0]) / np.sqrt(2),
    np.array([-1, 1, 0]) / np.sqrt(2),
    np.array([-1, 0, 0]),
    np.array([-1, -1, 0]) / np.sqrt(2),
    np.array([1, -1, 0]) / np.sqrt(2),
    np.array([1, 0, 0]),
]
phi_v = [
    np.pi,
    np.pi,
    -np.pi,
    np.pi,
    np.pi,
    -np.pi,
]

print(angle_mat(u_v, phi_v))
