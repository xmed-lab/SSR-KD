import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from utils import loader
from utils.geometry import Geometry



class ReconstructionDataset_unlabeled(Dataset):
    def __init__(self, split, num_repeat=None):
        # preset configs
        self.angles_options = loader.get_angles_options()
        self.npoint = 5000
        self.mask_ratio = 0.5
        self.geo = loader.load_projector()

        self.num_repeat = num_repeat
        self.id_list = loader.read_info()[split]

        self.data_list = []
        for obj_id in self.id_list:
            self.data_list.append({
                'obj_id': obj_id
            })

        if self.num_repeat is None:
            self.num_repeat = len(self.data_list)

        print(f'unlabeled: split {split}, num_repeat {self.num_repeat}, num_data {len(self.data_list)}')

    def __len__(self):
        return self.num_repeat

    def _sample_points(self):
        bb_max = np.array([+1, +1, +1])
        bb_min = -1 * bb_max
        points = np.random.rand(self.npoint, 3) * (bb_max - bb_min) + bb_min
        return points.astype(np.float32) # [N, 3] ~ [-1, +1]

    def __getitem__(self, index):
        index = index % len(self.data_list)
        data = deepcopy(self.data_list[index])

        # 1. load xrays
        angle_idx = np.random.randint(0, len(self.angles_options))
        angle_pair = self.angles_options[angle_idx]
        xrays = loader.read_xray(data['obj_id'], angle_pair)

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
        xrays_masked = xrays_masked.transpose(0, 3, 1, 2) # [M, 2, H, W]
        xrays_masked = xrays_masked * 6.
        angles = np.array(angles) # [M,]

        # 2. crop CT
        ct = loader.read_ct(data['obj_id'])
        crop_size = np.array([0.5, 0.5, 0.5])
        crop_start = np.random.rand(3) * (1 - crop_size)
        crop_end = crop_start + crop_size

        im_start = (crop_start * ct.shape).astype(int)
        im_end = (crop_end * ct.shape).astype(int)
        ct = ct[im_start[0]:im_end[0], im_start[1]:im_end[1], im_start[2]:im_end[2]]
        ct = ct.astype(np.float32) / 255.

        # 3. sample points
        points_ct = self._sample_points() # [N, 3]
        points = deepcopy(points_ct)
        points = (points + 1) * crop_size[None, :] + crop_start[None, :] * 2 - 1
        points = (points + 1) / 2
        points = points[:, [2, 1, 0]] # transpose is required as we do not apply transpose in the image loader (different from CT-recon codebse)
        
        points_proj = []
        for a in angles:
            p = self.geo.project(points, a)
            points_proj.append(p)
        points_proj = np.stack(points_proj, axis=0) # [M, N, 2]

        return {
            'ct': ct[None, ...],        # [1, H, W, D]
            'projs': xrays_masked,      # [M, 2, H, W]
            'points_proj': points_proj, # [M, N, 2]
            'points_ct': points_ct.T    # [3, N]
        }

