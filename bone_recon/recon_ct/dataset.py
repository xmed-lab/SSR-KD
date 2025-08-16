import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset

from utils import random_select, loader



class ReconstructionDataset_CT(Dataset):
    def __init__(self, split, npoint=5000, unlabeled=False, is_train=True, num_repeat=None, num_data=None):
        
        self.npoint = npoint
        self.unlabeled = unlabeled
        self.is_train = is_train
        self.num_repeat = num_repeat
        self.id_list = loader.read_info()[split]
        if num_data is not None:
            self.id_list = self.id_list[:num_data]
        
        self.data_list = []
        for obj_id in self.id_list:
            self.data_list.append({
                'id': obj_id,
                'ct': loader.read_ct(obj_id), # not normalized, [0, 255], uint8
                'sampling': loader.read_sampling(obj_id) if (not self.unlabeled) and self.is_train else None
            })

        if (not self.unlabeled) and self.is_train:
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

        print(f'labeled {not unlabeled}, split {split}, num_repeat {self.num_repeat}, num_data {len(self.data_list)}')

    def __len__(self):
        return self.num_repeat

    def _sample_points(self, sampling=None):
        if self.unlabeled:
            bb_max = np.array([+1, +1, +1])
            bb_min = -1 * bb_max
            points = np.random.rand(self.npoint, 3) * (bb_max - bb_min) + bb_min
            return points.astype(np.float32).T # [3, N] ~ [-1, +1]
        else:
            inside_points, outside_points = sampling['inside'], sampling['outside']
            inside_labels = sampling['inside_labels']

            in_npoint = self.npoint // 2
            inside_points, choice = random_select(inside_points, size=in_npoint)
            inside_labels, _ = random_select(inside_labels, choice=choice)
            outside_points, _ = random_select(outside_points, size=self.npoint - len(inside_points))

            points = np.concatenate([inside_points, outside_points], axis=0).T # [3, N]
            labels = np.concatenate([inside_labels, np.zeros([len(outside_points)])], axis=-1) # [N,]
            return points.astype(np.float32), labels.astype(np.float32)

    def __getitem__(self, index):
        index = index % len(self.data_list)
        data = deepcopy(self.data_list[index])
        ct = data['ct']
        sampling = data['sampling']

        if self.is_train:
            # 1. crop CT
            crop_size = np.array([0.5, 0.5, 0.5])
            crop_start = np.random.rand(3) * (1 - crop_size)
            crop_end = crop_start + crop_size

            im_start = (crop_start * ct.shape).astype(int)
            im_end = (crop_end * ct.shape).astype(int)
            ct = ct[im_start[0]:im_end[0], im_start[1]:im_end[1], im_start[2]:im_end[2]]

            # 2. flip CT
            flip_keys = np.random.choice([True, False], size=2)
            for i in range(2):
                if flip_keys[i]:
                    ct = np.flip(ct, axis=i + 1)

            # 3. sample points & rescale
            if self.unlabeled:
                points = self._sample_points()
            else:
                # crop
                p_start = crop_start * 2 - 1
                p_end = crop_end * 2 - 1
                for key in ['inside', 'outside']:
                    points = sampling[key]
                    crop_idx = (np.sum(points >= p_start, axis=1) == 3) & (np.sum(points <= p_end, axis=1) == 3)
                    sampling[key] = points[crop_idx, :]
                    if key == 'inside':
                        sampling['inside_labels'] = sampling['inside_labels'][crop_idx]

                points, labels = self._sample_points(sampling)
                points = (points.T - p_start) / (p_end - p_start) * 2 - 1 # crop-rescale
                for i in range(2):
                    if flip_keys[i]:
                        points[:, i + 1] *= -1 # flip
                points = points.astype(np.float32).T

        ct = ct[None, ...].astype(np.float32) / 255. # [1, H, W, D]
        if self.is_train:
            if self.unlabeled:
                return {
                    'ct': ct,
                    'points': points,
                }
            else:
                return {
                    'ct': ct,
                    'points': points,
                    'labels': labels
                }
        else:
            return {
                'id': data['id'],
                'ct': ct,
                'mesh_spacing': loader.read_spacing(data['id'])['mesh'],
                'submesh_path': loader.get_submesh_path(data['id'])
            }


if __name__ == '__main__':
    dst = ReconstructionDataset_CT(split='train')
    dst[0]
    pass
