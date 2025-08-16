import os
import csv
import trimesh
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils import loader
from utils.mesh import reconstruction, eval_mesh
from models.HGPIFuNet_xray import HGPIFuNet_xray
from recon_xray.dataset import ReconstructionDataset_Xray

import logging
logger = logging.getLogger('trimesh')
logger.setLevel(logging.ERROR)


geo = loader.load_projector()
angles = [a * np.pi / 180 for a in loader.get_angles_options()[0]]



def eval_func(points):
    points_proj = []
    for a in angles:
        points_tmp = points[[2, 1, 0], :].T
        points_tmp = (points_tmp + 1) / 2
        p = geo.project(points_tmp, a)
        points_proj.append(p)
    points_proj = np.stack(points_proj, axis=0) # [M, N, 2]
    points_proj = torch.from_numpy(points_proj[None, ...]).cuda().float()

    model.query(points_proj)
    pred = model.get_pred(part)[0, :] # [N,]
    pred = pred.data.cpu().numpy()
    pred[np.abs(points[0, :]) > 1.0] = -0.5
    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--name', type=str, default='semi+kd')
    parser.add_argument('--split', type=str, default='test_40')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--res', type=int, default=256)
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()

    # 1. load checkpoints
    save_dir = f'./recon_xray/logs/{args.name}'
    if args.epoch == -1:
        ckpt_path = os.path.join(save_dir, f'latest.pth')
        args.epoch = 'latest'
    else:
        ckpt_path = os.path.join(save_dir, f'epoch_{args.epoch}.pth')
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))

    # 2. load model
    model = HGPIFuNet_xray(in_ch=2, num_cls=5)
    model.load_state_dict(ckpt)
    model.cuda().eval()
    
    # 3. load data
    dst = ReconstructionDataset_Xray(split=args.split, is_train=False)
    loader = DataLoader(dst, pin_memory=True)

    # 4. evaluation
    save_dir = os.path.join(save_dir, f'results/ep_{args.epoch}')
    os.makedirs(save_dir, exist_ok=True)

    csv_file = open(os.path.join(save_dir, 'results.csv'), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['obj_id', '1', '2', '3', '4'])

    b_max = np.array([1 + 1e-2, 0.8, 0.8])
    b_min = -1 * b_max

    cd_list = []
    with torch.no_grad():
        for item in loader:
            obj_id = item['obj_id'][0]
            xray = item['projs'].float().cuda()

            model.encode_2d(xray)

            part_cds = []
            for part in range(1, 5):
                # get mesh prediction
                verts, faces = reconstruction(
                    args.res, 
                    b_min, b_max,
                    eval_func
                )
                mesh = trimesh.base.Trimesh(verts, faces)

                # postprocessing
                mesh_list = mesh.split(only_watertight=False)
                max_mesh = None
                max_cnt = -1
                for sub_mesh in mesh_list:
                    if len(sub_mesh.vertices) > max_cnt:
                        max_cnt = len(sub_mesh.vertices)
                        max_mesh = sub_mesh
                mesh = max_mesh
                mesh = trimesh.smoothing.filter_laplacian(mesh)

                # load gt mesh & evaluate
                gt_mesh = trimesh.load_mesh(item['submesh_path'][part - 1][0])
                spacing = item['mesh_spacing'][0].data.cpu().numpy()
                gt_mesh.vertices *= spacing
                mesh.vertices *= spacing

                cd = eval_mesh(gt_mesh, mesh)
                part_cds.append(cd)

                if args.save:
                    os.makedirs(os.path.join(save_dir, f'{obj_id}'), exist_ok=True)
                    mesh.export(os.path.join(save_dir, f'{obj_id}/sub_{part}.obj'))
            
            print(obj_id, part_cds)
            cd_list.append(part_cds)
            csv_writer.writerow([obj_id] + part_cds)

    cd_list = np.array(cd_list) # [N, 4]
    cd_mean = np.mean(cd_list, axis=0)
    print(cd_mean)

    csv_writer.writerow(['mean'] + list(cd_mean))
    csv_file.close()
