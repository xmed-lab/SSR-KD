from tqdm import tqdm
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

import utils



class SegmentationDataset(Dataset):
    def __init__(self, split='train', unlabeled=False, repeat_num=None):
        info = utils.read_info()

        self.repeat_num = repeat_num
        self.unlabeled = unlabeled

        splits = split.split('+')
        self.id_list = []
        for s in splits:
            self.id_list += info[s]

        if self.repeat_num is None:
            self.repeat_num = len(self.id_list)

        self.transform = A.Compose([
            A.RandomCrop(width=384, height=384),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])

        print(f'dataset (base) initialization: split {split}, data {len(self.id_list)}, repeat {self.repeat_num}, unlabeled {unlabeled}')

    def __len__(self):
        return self.repeat_num

    def __getitem__(self, index):
        index = index % len(self.data_list)
        data = self.data_list[index]
        obj_id = data['obj_id']
        image = data['image']
        mask = data['seg_mask']
        angle = data['angle']

        # augmentation
        if not self.unlabeled:
            aug_data = self.transform(image=image, mask=mask)
            image, mask = aug_data['image'], aug_data['mask']
        else:
            image = self.transform(image=image)['image']

        # collect data
        ret_dict = {
            'image': image.transpose(2, 0, 1) # [1, H, W]
        }
        
        if not self.unlabeled:
            ret_dict['seg_mask'] = mask[..., 0] # [H, W]
        
        return ret_dict


class SegmentationDataset_test(Dataset):
    def __init__(self, obj_id):
        self.data_list = []
        images, masks, angles = utils.read_data(obj_id)
        for i in range(len(images)):
            self.data_list.append({
                'obj_id': obj_id,
                'image': images[i][..., None],
                'seg_mask': masks[i][..., None] if masks is not None else None,
                'angle': angles[i]
            })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        image = data['image']
        mask = data['seg_mask']

        # collect data
        ret_dict = {
            'image': image.transpose(2, 0, 1) # [1, H, W]
        }
        
        if mask is not None:
            ret_dict['seg_mask'] = mask[..., 0] # [H, W]
        
        ret_dict['obj_id'] = data['obj_id']
        ret_dict['angle'] = data['angle']
        
        return ret_dict
