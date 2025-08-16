import os
import yaml
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from skimage.measure import label, regionprops
from projector import Projector, visualize_projections



if __name__ == '__main__':
    data_dir = './processed'

    angles_degree = np.linspace(0, 360, 37).astype(int)
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for obj_id in tqdm(sorted(os.listdir(data_dir))):
        mask_path = os.path.join(data_dir, obj_id, 'seg_mask.nii.gz')
        if os.path.exists(mask_path):
            seg_mask = sitk.ReadImage(mask_path)
            seg_mask = sitk.GetArrayFromImage(seg_mask)
            seg_mask = seg_mask.astype(np.float32)
            seg_mask = seg_mask.transpose(2, 1, 0)

            projector = Projector(config, angles_degree)
            projs = projector(seg_mask)

            angles = projs['angles']
            projs = (projs['projs'] > 0).astype(int)
            
            for proj in projs:
                labeled_mask = label(proj)
                regions = regionprops(labeled_mask)
                
                areas = [region.area for region in regions]
                
                if len(areas) > 0:
                    area_threshold = max(np.mean(areas) * 0.1, 10)
                    
                    cleaned_mask = np.zeros_like(proj)
                    for region in regions:
                        if region.area >= area_threshold:
                            cleaned_mask[labeled_mask == region.label] = 1
                    
                    proj[:] = cleaned_mask

            visualize_projections(os.path.join(data_dir, obj_id, 'bone_masks.png'), projs, angles)
            np.savez(os.path.join(data_dir, obj_id, 'bone_masks.npz'), projs=projs, angles=angles)
