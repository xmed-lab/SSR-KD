import os
import json
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from scipy.ndimage import zoom


def norm_ct(image):
    image = np.clip(image, a_min=-400, a_max=500)
    image = (image - image.min()) / (image.max() - image.min())
    image *= 255.
    image = image.astype(np.uint8)
    return image


if __name__ == '__main__':
    save_dir = './processed'
    data_dir = '../../dataset_zhaowei/'

    info = json.load(open('info.json'))
    obj_ids = info['train'] + info['unlabeled'] + info['test'] + info['eval']
    obj_ids = list(set(obj_ids))

    for obj_id in tqdm(sorted(obj_ids)):
        for tag in ['FL', 'FR', 'ML', 'MR']:
            path = os.path.join(data_dir, f'{tag}/{obj_id}.mhd')
            if os.path.exists(path):
                itk_img = sitk.ReadImage(path)
                
                original_spacing = itk_img.GetSpacing()
                original_size = itk_img.GetSize()
                
                new_spacing = [
                    original_spacing[0] * original_size[0] / 512,
                    original_spacing[1] * original_size[1] / 512,
                    original_spacing[2] * original_size[2] / 512
                ]
                
                resample_filter = sitk.ResampleImageFilter()
                resample_filter.SetSize((512, 512, 512))
                resample_filter.SetOutputSpacing(new_spacing)
                resample_filter.SetOutputOrigin(itk_img.GetOrigin())
                resample_filter.SetOutputDirection(itk_img.GetDirection())
                resample_filter.SetInterpolator(sitk.sitkLinear)
                resample_filter.SetDefaultPixelValue(-2048)
                resample_filter.SetOutputPixelType(itk_img.GetPixelID())
                itk_img = resample_filter.Execute(itk_img)

                image = sitk.GetArrayFromImage(itk_img) # -1, 512, 512
                image = norm_ct(image)

                obj_dir = os.path.join(save_dir, obj_id)
                os.makedirs(obj_dir, exist_ok=True)
                sitk.WriteImage(sitk.GetImageFromArray(image), os.path.join(obj_dir, 'ct_512x.nii.gz'))

                break
