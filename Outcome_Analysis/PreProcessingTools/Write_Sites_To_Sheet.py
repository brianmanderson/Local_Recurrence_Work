__author__ = 'Brian M Anderson'
# Created on 1/26/2021
import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool.ReaderWriter import DicomReaderWriter, pydicom, plot_scroll_Image


def write_sites_to_excel(nifti_export_path, excel_path):
    stats = sitk.LabelShapeStatisticsImageFilter()
    Connected_Component_Filter = sitk.ConnectedComponentImageFilter()
    df = pd.read_excel(excel_path, engine='openpyxl')
    unfilled = df.loc[(pd.isnull(df['Non_Recurrence_Sites'])) | (pd.isnull(df['Recurrence_Sites']))]
    if unfilled.shape[0] > 0:
        for index in unfilled.index:
            patient_id = unfilled['PatientID'][index]
            print('Running on {}'.format(patient_id))
            mask_handle = sitk.ReadImage(os.path.join(nifti_export_path, '{}_Primary_Mask.nii'.format(patient_id)))
            mask = sitk.GetArrayFromImage(mask_handle)
            status_3 = mask == 1
            num_status_3 = 0
            if np.max(status_3) > 0:
                status_3_image = sitk.GetImageFromArray(status_3.astype('int'))
                connected_image = Connected_Component_Filter.Execute(status_3_image)
                stats.Execute(connected_image)
                num_status_3 = len(stats.GetLabels())
            df.at[index, 'Non_Recurrence_Sites'] = num_status_3
            status_1_or_2 = mask == 2
            num_status_1_or_2 = 0
            if np.max(status_1_or_2) > 0:
                status_1_or_2_image = sitk.GetImageFromArray(status_1_or_2.astype('int'))
                connected_image = Connected_Component_Filter.Execute(status_1_or_2_image)
                stats.Execute(connected_image)
                num_status_1_or_2 = len(stats.GetLabels())
            df.at[index, 'Recurrence_Sites'] = num_status_1_or_2
        df.to_excel(excel_path, index=0)
    return None


if __name__ == '__main__':
    pass
