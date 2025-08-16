
import trimesh
import numpy as np
from skimage import measure

import torch

from utils.chamfer import chamfer_distance



def sample_points(mesh, npoint):
    points, _ = trimesh.sample.sample_surface_even(mesh, npoint)
    points = torch.from_numpy(points).float().cuda()
    points = points.unsqueeze(0)
    return points


def eval_mesh(ref_mesh, src_mesh, npoint=8192 * 6):
    ref_points = sample_points(ref_mesh, npoint)
    src_points = sample_points(src_mesh, npoint)
    dist_ref, dist_src = chamfer_distance(ref_points, src_points, pointwise=True)
    dist_ref = dist_ref[0].data.cpu().numpy()
    dist_src = dist_src[0].data.cpu().numpy()
    cd = (dist_ref.mean() + dist_src.mean()) / 2.0
    return cd


def reconstruction(resolution, b_min, b_max, eval_func, num_samples=50000):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(
        resolution, resolution, resolution,
        b_min, b_max, 
        transform=None
    ) # coords: [3, H, W, D]

    # Then we evaluate the grid
    sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    verts, faces, _, _ = measure.marching_cubes(sdf, 0.0)
    
    # transform verts into world coordinate system
    verts = np.matmul(mat[:3, :3], verts.T) + mat[:3, 3:4]
    verts = verts.T
    return verts, faces


def create_grid(resX, resY, resZ, b_min=np.array([0, 0, 0]), b_max=np.array([1, 1, 1]), transform=None):
    '''
    Create a dense grid of given resolution and bounding box
    :param resX: resolution along X axis
    :param resY: resolution along Y axis
    :param resZ: resolution along Z axis
    :param b_min: vec3 (x_min, y_min, z_min) bounding box corner
    :param b_max: vec3 (x_max, y_max, z_max) bounding box corner
    :return: [3, resX, resY, resZ] coordinates of the grid, and transform matrix from mesh index
    '''
    coords = np.mgrid[:resX, :resY, :resZ]
    coords = coords.reshape(3, -1)
    coords_matrix = np.eye(4)
    length = b_max - b_min
    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ
    coords_matrix[:3, 3] = b_min
    coords = np.matmul(coords_matrix[:3, :3], coords) + coords_matrix[:3, 3:4]
    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    return coords, coords_matrix


def eval_grid(coords, eval_func, num_samples):
    resolution = coords.shape[1:4]
    coords = coords.reshape([3, -1])
    sdf = batch_eval(coords, eval_func, num_samples=num_samples)
    return sdf.reshape(resolution)


def batch_eval(points, eval_func, num_samples):
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = eval_func(
            points[:, i * num_samples:i * num_samples + num_samples])
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = eval_func(points[:, num_batches * num_samples:])

    return sdf


def save_obj_mesh(mesh_path, verts, faces):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()
