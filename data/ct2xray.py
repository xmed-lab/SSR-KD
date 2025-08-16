import os
import yaml
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from projector import Projector, visualize_projections


if __name__ == '__main__':
    data_dir = './processed'

    angles = np.linspace(0, 360, 37).astype(int)
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for obj_id in tqdm(sorted(os.listdir(data_dir))):
        ct_path = os.path.join(data_dir, obj_id, 'ct_512x.nii.gz')
        ct = sitk.ReadImage(ct_path)
        image = sitk.GetArrayFromImage(ct)
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 1, 0)

        projector = Projector(config, angles)
        projs = projector(image)

        visualize_projections(os.path.join(data_dir, obj_id, 'projs.png'), projs['projs'], projs['angles'])
        np.savez(os.path.join(data_dir, obj_id, 'projs.npz'), projs=projs['projs'], angles=projs['angles'])
