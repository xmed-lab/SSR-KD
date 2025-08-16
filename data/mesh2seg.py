import os
import trimesh
import numpy as np
import open3d as o3d
import SimpleITK as sitk
from tqdm import tqdm


def mesh2sdf(
        mesh_dir, 
        res=512
    ):

    grid_points = np.mgrid[:res, :res, :res]
    grid_points = grid_points.reshape(3, -1).transpose(1, 0).astype(np.float32)
    grid_points = grid_points * 2 / res - 1

    seg_mask = np.zeros((res, res, res), dtype=np.uint8)
    for part in range(4):
        mesh_path = os.path.join(mesh_dir, f'sub_{part}.obj')
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices)
        triangels = np.asarray(mesh.triangles)

        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        device = o3d.core.Device('CPU:0')

        mesh = o3d.t.geometry.TriangleMesh(device=device)
        mesh.vertex.positions = o3d.core.Tensor(vertices, dtype_f, device)
        mesh.triangle.indices = o3d.core.Tensor(triangels, dtype_i, device)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        points = o3d.core.Tensor(grid_points, dtype_f, device)
        occ = scene.compute_occupancy(points).numpy()
        occ = occ.reshape(res, res, res)
        seg_mask[occ == 1] = part + 1
    
    return seg_mask


if __name__ == '__main__':
    data_dir = './processed'
    
    for obj_id in tqdm(sorted(os.listdir(data_dir))):
        if os.path.exists(os.path.join(data_dir, obj_id, 'mesh.obj')):
            mesh_dir = os.path.join(data_dir, obj_id, 'submesh')
            occ = mesh2sdf(mesh_dir, res=512)
            sitk.WriteImage(sitk.GetImageFromArray(occ), os.path.join(data_dir, obj_id, 'seg_mask.nii.gz'))
        else:
            print(f'No mesh found for {obj_id}')
