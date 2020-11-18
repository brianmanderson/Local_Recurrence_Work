__author__ = 'Brian M Anderson'
# Created on 11/18/2020
import os
import SimpleITK as sitk
import numpy as np


def check_liver_volume(nifti_export_path):
    primary_masks = [i for i in os.listdir(nifti_export_path) if i.endswith('Primary_Mask.nii')]
    for file in primary_masks:
        primary_mask = sitk.ReadImage(os.path.join(nifti_export_path, file))
        secondary_mask = sitk.ReadImage(os.path.join(nifti_export_path, file.replace('Primary', 'Secondary')))
        primary_mask_array = sitk.GetArrayFromImage(primary_mask)
        secondary_mask_array = sitk.GetArrayFromImage(secondary_mask)
        primary_volume = np.prod(primary_mask.GetSpacing()) * np.sum(primary_mask_array > 0) / 1000  # in cc
        secondary_volume = np.prod(secondary_mask.GetSpacing()) * np.sum(secondary_mask_array > 0) / 1000  # in cc
        volume_change = np.abs(primary_volume - secondary_volume) / np.min([primary_volume, secondary_volume]) * 100
        if primary_volume < 500 or secondary_volume < 500:
            print('Might want to check out {}'.format(file))
        elif volume_change > 30:
            print('Might want to check out {}, {}% volume change'.format(file, volume_change))
    return None


if __name__ == '__main__':
    pass
