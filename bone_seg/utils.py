import os
import cv2
import json
import numpy as np
from torch import nn



data_root = '../data/'


def read_info():
    file_path = os.path.join(data_root, 'info.json')
    with open(file_path) as f:
        info = json.load(f)
    return info


def read_data(obj_id):
    data_dir = os.path.join(data_root, 'processed')

    projs = np.load(os.path.join(data_dir, obj_id, 'projs.npz'))
    images = projs['projs']
    angles = projs['angles']
    mask_path = os.path.join(data_dir, obj_id, 'bone_masks.npz')
    if os.path.exists(mask_path):
        masks = np.load(mask_path)['projs']
    else:
        masks = None

    return images, masks, angles


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(weight, epoch, max_epoch):
    return weight * sigmoid_rampup(epoch, max_epoch)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight)
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
    return model
