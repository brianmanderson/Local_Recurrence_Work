__author__ = 'Brian M Anderson'
# Created on 11/11/2020
import os
import pandas as pd
from Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, pydicom, plot_scroll_Image
import SimpleITK as sitk
import numpy as np
from Local_Recurrence_Work.Outcome_Analysis.RegisteringImages.src.RegisterImages.WithDicomReg import \
    register_images_with_dicom_reg


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    MRN_list, GTV_List, Ablation_list, Registered_list = df['MRN'].values, df['PreExam'].values,\
                                                         df['Ablation_Exam'].values, df['Registered'].values
    MRN_dictionary = {}
    for MRN, GTV, Ablation, Registered in zip(MRN_list, GTV_List, Ablation_list, Registered_list):
        Registered = str(Registered)
        if Registered != '1.0':
            continue
        add = True
        if type(GTV) is float or type(Ablation) is float:
            add = False
        if add:
            GTV = str(GTV)
            if GTV.startswith('CT'):
                if GTV.find(' ') == -1:
                    GTV = 'CT {}'.format(GTV.split('CT')[-1])
            Ablation = str(Ablation)
            if Ablation.startswith('CT'):
                if Ablation.find(' ') == -1:
                    Ablation = 'CT {}'.format(Ablation.split('CT')[-1])
            MRN_dictionary[MRN] = {'Primary': GTV, 'Secondary': Ablation}
    return MRN_dictionary


def main():
    base_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'

    MRN_dictionary = return_MRN_dictionary(excel_path)
    for MRN in MRN_dictionary.keys():
        if MRN not in os.listdir(base_export_path):
            continue
        """
        For each patient, load in the primary and secondary DICOM images, also liver.
        On primary, pull in the Retro_GTV, Recurred, these shouldn't be present on the secondary
        """
        patient_dictionary = MRN_dictionary[MRN]
        primary = patient_dictionary['Primary']
        secondary = patient_dictionary['Secondary']
        assocations = {'Liver_BMA_Program_4': 'Liver'}
        primary_reader = DicomReaderWriter(Contour_Names=['Retro_GTV', 'Retro_GTV_Recurred', 'Liver'],  # Need all
                                           associations=assocations, arg_max=False, require_all_contours=False)
        secondary_reader = DicomReaderWriter(Contour_Names=['Liver'], associations=assocations)  # Only need the liver
        for root, directories, files in os.walk(os.path.join(base_export_path, MRN)):
            if 'Registration' in directories and primary in directories and secondary in directories:
                '''
                Load in our registration
                '''
                registration_path = os.path.join(root, 'Registration')
                registration_file = [os.path.join(registration_path, i) for i in os.listdir(registration_path)][0]
                dicom_registration = pydicom.read_file(registration_file)
                '''
                Next, our primary and secondary images, as sitkFloat32
                '''
                primary_path = os.path.join(root, primary)
                secondary_path = os.path.join(root, secondary)
                primary_reader.down_folder(primary_path)
                secondary_reader.down_folder(secondary_path)
                fixed_dicom_image = sitk.Cast(primary_reader.dicom_handle, sitk.sitkFloat32)
                moving_dicom_image = sitk.Cast(secondary_reader.dicom_handle, sitk.sitkFloat32)
                moving_dicom_mask = sitk.Cast(secondary_reader.annotation_handle, sitk.sitkFloat32)
                """
                Resample the moving image to register with the primary
                """
                resampled_moving_image = register_images_with_dicom_reg(fixed_image=fixed_dicom_image,
                                                                        moving_image=moving_dicom_image,
                                                                        dicom_registration=dicom_registration,
                                                                        min_value=-1000)
                """
                Resample the liver contour as well
                """
                resampled_moving_mask = register_images_with_dicom_reg(fixed_image=fixed_dicom_image,
                                                                       moving_image=moving_dicom_mask,
                                                                       dicom_registration=dicom_registration,
                                                                       min_value=0)
                resampled_moving_mask = sitk.GetArrayFromImage(resampled_moving_mask)
                resampled_moving_mask[resampled_moving_mask > 0] = 1
                resampled_moving_mask = sitk.GetImageFromArray(resampled_moving_mask.astype('int8'))
                resampled_moving_mask.SetOrigin(resampled_moving_image.GetOrigin())
                resampled_moving_mask.SetDirection(resampled_moving_image.GetDirection())
                resampled_moving_mask.SetSpacing(resampled_moving_image.GetSpacing())
                xxx = 1


if __name__ == "__main__":
    main()
