__author__ = 'Brian M Anderson'
# Created on 11/20/2020
import os
import pandas as pd
import SimpleITK as sitk
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image


def create_secondary_deformed_nifti(nifti_export_path, deformation_export_path, anonymized_sheet):
    patient_df = pd.read_excel(anonymized_sheet)
    deformed_image_files = [i for i in os.listdir(deformation_export_path) if i.endswith('.mhd')]
    for deformed_image in deformed_image_files:
        print(deformed_image)
        split_up = deformed_image.strip('.mhd').split('_')
        if len(split_up) == 4:
            MRN, primary, _, secondary = split_up
        else:
            MRN = '{}_{}_{}'.format(split_up[0], split_up[1], split_up[2])
            primary, secondary = split_up[3], split_up[5]
            xxx = 1
        inside_df = patient_df.loc[(patient_df['MRN'].astype('str') == MRN) & (patient_df['PreExam'] == primary) &
                                   (patient_df['PostExam'] == secondary)]
        assert inside_df.shape[0] <= 1, 'Somethings wrong here..{}'.format(MRN)
        if inside_df.shape[0] == 1:
            patient_id = int(inside_df['PatientID'].values[0])
            out_path = os.path.join(nifti_export_path, '{}_Secondary_Deformed.nii'.format(patient_id))
        else:
            print('Not found here...{}'.format(MRN))
            continue
        if os.path.exists(out_path):
            continue
        deformed_handle = sitk.ReadImage(os.path.join(deformation_export_path, deformed_image))
        primary_handle = sitk.ReadImage(os.path.join(nifti_export_path, '{}_Primary_Dicom.nii'.format(patient_id)))
        if deformed_handle.GetSize() != primary_handle.GetSize():
            # print('These are not the same for {}...'.format(MRN))
            deformed_handle = sitk.Resample(deformed_handle, primary_handle, sitk.AffineTransform(3), sitk.sitkLinear,
                                            -1000, deformed_handle.GetPixelID())
        deformed_handle.SetSpacing(primary_handle.GetSpacing())
        sitk.WriteImage(deformed_handle, out_path)
    return None


if __name__ == '__main__':
    pass
