import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from dataset import SegmentationDataset_test
from models.unet_model import UNet
from utils import read_info


def visualize_projections(path, projections, angles, figs_per_row=10):
    n_row = np.ceil(len(projections) / figs_per_row).astype(int)
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())

    for i in range(len(projections)):
        angle = int((angles[i] / np.pi) * 180)
        plt.subplot(n_row, figs_per_row, i + 1)
        plt.imshow(projections[i] * 255, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{angle}')
        plt.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(path, dpi=500)
    plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--epoch', type=int, default=-1)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--save', action='store_true', default=False)
    args = parser.parse_args()


    info = read_info()
    if args.split == 'all':
        id_list = []
        for k, v in info.items():
            id_list += v
    else:
        splits = args.split.split('+')
        id_list = []
        for s in splits:
            id_list += info[s]
    id_list = sorted(list(set(id_list)))

    save_dir = f'./logs/{args.name}'
    if args.epoch == -1:
        ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
        args.epoch = 'latest'
    else:
        ckpt = torch.load(os.path.join(save_dir, f'epoch_{args.epoch}.pth'))
    
    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(ckpt['A'])
    model.cuda().eval()

    if args.save:
        save_dir = os.path.join(save_dir, 'results', f'ep_{args.epoch}')
        os.makedirs(save_dir, exist_ok=True)

    iou_list = []
    for obj_id in tqdm(id_list):
        dst = SegmentationDataset_test(obj_id)
        loader = DataLoader(dst, pin_memory=True)

        pseudo_masks = []
        angles = []
        with torch.no_grad():
            for data in loader:
                data['image'] = data['image'].cuda()
                logit = model(data['image'])
                seg_pred = torch.argmax(logit, dim=1)
                seg_pred = seg_pred[0].data.cpu().numpy()
                pseudo_masks.append(seg_pred)
                angles.append(float(data['angle'][0].data.cpu().numpy()))

                if 'seg_mask' in data.keys():
                    seg_mask = data['seg_mask'][0].data.cpu().numpy().astype(int)
                    inter = np.sum((seg_mask == 1) & (seg_pred == 1))
                    union = np.sum((seg_mask + seg_pred) > 0)
                    iou = inter / (union + 1e-10)
                    iou_list.append(iou)

            if args.save:
                pseudo_masks = np.stack(pseudo_masks, axis=0).astype(np.uint8)
                angles = np.stack(angles, axis=0)
                os.makedirs(os.path.join(save_dir, obj_id), exist_ok=True)
                np.savez(os.path.join(save_dir, obj_id, 'bone_masks.npz'), projs=pseudo_masks, angles=angles)
                visualize_projections(os.path.join(save_dir, obj_id, 'bone_masks.png'), pseudo_masks, angles)

    print('iou:', np.mean(iou_list))
