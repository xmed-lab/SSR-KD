import cv2
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from utils import random_select, loader



class ReconstructionDataset_Xray(Dataset):
    def __init__(self, split, is_train=True, num_repeat=None):
        # preset configs
        self.angles_options = loader.get_angles_options()
        self.min_z = 0.7
        self.npoint = 5000
        self.mask_ratio = 0.5
        self.geo = loader.load_projector()

        self.is_train = is_train
        self.num_repeat = num_repeat
        self.id_list = loader.read_info()[split]

        self.data_list = []
        for obj_id in self.id_list:
            self.data_list.append({
                'obj_id': obj_id,
                'sampling': loader.read_sampling(obj_id) if self.is_train else None # normalized [-1, +1] (only points)
            })

        if self.is_train:
            count = np.zeros(4)
            for item in self.data_list:
                labels = item['sampling']['inside_labels']
                for i in range(4):
                    count[i] += np.sum(labels == i + 1)

            count /= np.sum(count)
            self.weight = np.ones(4 + 1)
            self.weight[1:] = np.power(np.amax(count) / count, 1/3)
            self.weight[0] = 1.5
            print('weights:', self.weight)

        if self.num_repeat is None:
            self.num_repeat = len(self.data_list)

        print(f'labeled: split {split}, num_repeat {self.num_repeat}, num_data {len(self.data_list)}')

    def __len__(self):
        return self.num_repeat
    
    def _sample_points(self, sampling):
        inside_points, outside_points = sampling['inside'], sampling['outside']
        inside_labels = sampling['inside_labels']

        in_npoint = self.npoint // 2
        inside_points, choice = random_select(inside_points, size=in_npoint)
        inside_labels, _ = random_select(inside_labels, choice=choice)
        outside_points, _ = random_select(outside_points, size=self.npoint - len(inside_points))

        points = np.concatenate([inside_points, outside_points], axis=0)                   # [N, 3]
        labels = np.concatenate([inside_labels, np.zeros([len(outside_points)])], axis=-1) # [N,]
        return points.astype(np.float32), labels.astype(np.float32)

    def __getitem__(self, index):
        index = index % len(self.data_list)
        data = deepcopy(self.data_list[index])
        obj_id = data['obj_id']
        sampling = data['sampling']

        # 1. load xrays
        angle_idx = np.random.randint(0, len(self.angles_options)) if self.is_train else 0
        angle_pair = self.angles_options[angle_idx]
        xrays = loader.read_xray(obj_id, angle_pair)

        xrays_masked = []
        angles = []
        for xray_info in xrays:
            xray = xray_info['xray']
            mask = xray_info['mask'].astype(np.float32)
            masked = (mask + (1 - mask) * self.mask_ratio) * xray
            masked = np.stack([masked, xray], axis=-1) # [H, W, 2]
            xrays_masked.append(masked)
            angles.append(xray_info['angle'])

        xrays_masked = np.stack(xrays_masked, axis=0) # [M, H, W, 2]
        xrays_masked = xrays_masked * 6.
        angles = np.array(angles) # [M,]

        # 2. z-crop
        if self.is_train:
            # apply z-crop
            z_size = np.random.rand() * (1 - self.min_z) + self.min_z
            z_start = np.random.rand() * (1 - z_size)

            # 2.1 crop xrays
            im_res = xrays_masked.shape[1]
            im_start = int(im_res * z_start)
            im_end = im_start + int(im_res * z_size)
            xrays_masked = xrays_masked[:, im_start:im_end, ...] # [M, H', W, 2]
            h, w = xrays_masked.shape[1:3]
            xrays_masked = xrays_masked.transpose(1, 2, 3, 0).reshape(h, w, -1) # [H', W, 2M]
            xrays_masked = cv2.resize(xrays_masked, (im_res, im_res))
            xrays_masked = xrays_masked.reshape(im_res, im_res, 2, -1).transpose(3, 2, 0, 1) # [M, 2, H, W]

            # 2.2 crop sampling
            # sampling - mesh - ct, aligned
            # xray (z-flipped) due to the projection mechanism
            p_end = 1 - z_start * 2
            p_start = p_end - z_size * 2
            for key in ['inside', 'outside']:
                points = sampling[key]
                choice = (points[:, 0] >= p_start) & (points[:, 0] <= p_end)
                sampling[key] = points[choice]
                if key == 'inside':
                    sampling['inside_labels'] = sampling['inside_labels'][choice]

            # 2.3 sample points
            points, labels = self._sample_points(sampling)  # [N, 3]
            points[:, 0] = (points[:, 0] - p_start) / (p_end - p_start) # rescale and map to [0, 1]
            points[:, 1:3] = (points[:, 1:3] + 1) / 2 # map to [0, 1]
            points = points[:, [2, 1, 0]]
            
            # 2.4 compute projected 2D coordinates
            points_proj = []
            for a in angles:
                p = self.geo.project(points, a)
                points_proj.append(p)
            points_proj = np.stack(points_proj, axis=0) # [M, N, 2]

            return {
                'projs': xrays_masked,      # [M, 2, H, W]
                'points_proj': points_proj, # [M, N, 2]
                'points_gt': labels         # [N]
            }
        else:
            return {
                'obj_id': obj_id,
                'projs': xrays_masked.transpose(0, 3, 1, 2), # [M, 2, H, W]
                'mesh_spacing': loader.read_spacing(obj_id)['mesh'],
                'submesh_path': loader.get_submesh_path(obj_id)
            }
