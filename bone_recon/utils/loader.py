import os
import yaml
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom

from utils.geometry import Geometry



data_root = '../data/processed/'
seg_root = '../bone_seg/logs/default/results/ep_200/'



def get_angles_options():
    angles_options = [
        [0,    90],
        [0,   270],
        [180,  90],
        [180, 270]
    ]
    return angles_options


def read_info():
    file_path = os.path.join(data_root, '../info.json')
    with open(file_path) as f:
        info = json.load(f)
    info['test'] = sorted(info['test'])
    return info # keys: ['train', 'unlabeled', 'test'], [N,] (str)


def load_projector():
    with open(os.path.join(data_root, '../config.yaml')) as f:
        config = yaml.safe_load(f)
    projector = Geometry(config)
    return projector


def read_xray(obj_id, angle_pair):
    projs = np.load(os.path.join(data_root, f'{obj_id}/projs.npz'))
    seg_masks = np.load(os.path.join(seg_root, f'{obj_id}/bone_masks.npz'))
    
    angles = projs['angles']
    angles_check = seg_masks['angles']
    if not np.sum(angles == angles_check) == len(angles):
        raise ValueError(f'angles mismatch: {obj_id}')

    ret_info = []
    for a in angle_pair:
        for i in range(len(angles)):
            a_tmp = np.round(angles[i] * 180 / np.pi)
            a_tmp = int(a_tmp)
            if a_tmp == a:
                ret_info.append({
                    'xray': projs['projs'][i],     # float32
                    'mask': seg_masks['projs'][i], # uint8
                    'angle': angles[i]
                })
                break
    
    if len(ret_info) != len(angle_pair):
        raise ValueError(f'angle pair not found: {obj_id}, {angle_pair}')
    
    return ret_info


def read_ct(obj_id):
    path = os.path.join(data_root, f'{obj_id}/ct_256x.nii.gz')
    itk_img = sitk.ReadImage(path)
    itk_arr = sitk.GetArrayFromImage(itk_img)
    return itk_arr # [H, W, D]


def read_sampling(obj_id):
    path = os.path.join(data_root, f'{obj_id}/sampling.npz')
    sampling = np.load(path, allow_pickle=True)['arr_0'][None][0]
    inside_points = np.concatenate([sampling['surface_inside'], sampling['random_inside']], axis=0)
    outside_points = np.concatenate([sampling['surface_outside'], sampling['random_outside']], axis=0)
    inside_labels = np.concatenate([sampling['surface_inside_labels'], sampling['random_inside_labels']], axis=0)
    return {
        'inside': inside_points,        # [N, 3]
        'inside_labels': inside_labels, # [N,]
        'outside': outside_points       # [M, 3]
    }


def read_spacing(obj_id):
    path = os.path.join(data_root, f'{obj_id}/spacing.npz')
    spacing = np.load(path, allow_pickle=True)
    spacing = dict(spacing)
    return spacing # keys: ['mesh', 'ct', 'xray_frontal', 'xray_lateral'], [d,]


def get_mesh_path(obj_id):
    return os.path.join(data_root, f'{obj_id}/mesh.obj')


def get_submesh_path(obj_id):
    res = []
    for i in range(4):
        res.append(os.path.join(data_root, f'{obj_id}/submesh/sub_{i}.obj'))
    return res
