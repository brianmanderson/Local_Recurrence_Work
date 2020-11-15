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
            MRN_dictionary[str(MRN)] = {'Primary': GTV, 'Secondary': Ablation}
    return MRN_dictionary


def register_images_to_nifti(dicom_export_path, nifti_export_path, excel_path, anonymized_sheet):
    patient_df = pd.read_excel(anonymized_sheet)
    MRN_dictionary = return_MRN_dictionary(excel_path)
    for MRN in MRN_dictionary.keys():
        if MRN not in os.listdir(dicom_export_path):
            print('{} not present in collection'.format(MRN))
            continue
        print(MRN)
        """
        Check and see if we've done this one before
        """
        add_patient = True
        if MRN in patient_df['MRN'].values:
            patient_id = int(patient_df['PatientID'].values[list(patient_df['MRN'].values).index(MRN)])
            add_patient = False
            if os.path.exists(os.path.join(nifti_export_path, '{}_Primary_Dicom.nii'.format(patient_id))):
                continue
        else:
            patient_id = 0
            int_ids = [int(i) for i in patient_df['PatientID'].values]
            while patient_id in int_ids:
                patient_id += 1
        """
        For each patient, load in the primary and secondary DICOM images, also liver.
        On primary, pull in the Retro_GTV, Recurred, these shouldn't be present on the secondary
        """
        patient_dictionary = MRN_dictionary[MRN]
        primary = patient_dictionary['Primary']
        secondary = patient_dictionary['Secondary']
        assocations = {'Liver_BMA_Program_4': 'Liver'}
        primary_reader = DicomReaderWriter(Contour_Names=['Retro_GTV', 'Retro_GTV_Recurred', 'Liver'],
                                           associations=assocations, require_all_contours=False)
        secondary_reader = DicomReaderWriter(Contour_Names=['Liver'], associations=assocations)  # Only need the liver
        for root, directories, files in os.walk(os.path.join(dicom_export_path, MRN)):
            if 'Registration' in directories and primary in directories and secondary in directories:
                '''
                Load in our registration
                '''
                registration_path = os.path.join(root, 'Registration')
                registration_file = [os.path.join(registration_path, i) for i in os.listdir(registration_path)][0]
                if registration_file.endswith('.dcm'):
                    dicom_registration = pydicom.read_file(registration_file)
                else:
                    dicom_registration = None
                '''
                Next, our primary and secondary images, as sitkFloat32
                '''
                primary_path = os.path.join(root, primary)
                secondary_path = os.path.join(root, secondary)
                primary_reader.down_folder(primary_path)
                has_liver = False
                for roi in primary_reader.rois_in_case:
                    if roi in primary_reader.associations.keys() and primary_reader.associations[roi] == 'liver':
                        has_liver = True
                        break
                if not has_liver:
                    print('No liver contours at {}'.format(primary_path))
                    continue
                else:
                    has_liver = False
                    secondary_reader.down_folder(secondary_path)
                    for roi in secondary_reader.rois_in_case:
                        if roi in secondary_reader.associations.keys() and \
                                secondary_reader.associations[roi] == 'liver':
                            has_liver = True
                            break
                if not has_liver:
                    print('No liver contours at {}'.format(secondary_path))
                    continue

                fixed_dicom_image = sitk.Cast(primary_reader.dicom_handle, sitk.sitkFloat32)
                fixed_dicom_mask = primary_reader.annotation_handle
                moving_dicom_image = sitk.Cast(secondary_reader.dicom_handle, sitk.sitkFloat32)
                moving_dicom_mask = sitk.Cast(secondary_reader.annotation_handle, sitk.sitkFloat32)
                """
                Resample the moving image to register with the primary
                """
                resampled_moving_image = register_images_with_dicom_reg(fixed_image=fixed_dicom_image,
                                                                        moving_image=moving_dicom_image,
                                                                        dicom_registration=dicom_registration,
                                                                        min_value=-1000, method=sitk.sitkLinear)
                """
                Resample the liver contour as well, we're using a linear sampling here because we often have large 
                rotations present.. cut off of 0.5 seemed appropriate
                """
                resampled_moving_mask = register_images_with_dicom_reg(fixed_image=fixed_dicom_mask,
                                                                       moving_image=moving_dicom_mask,
                                                                       dicom_registration=dicom_registration,
                                                                       min_value=0, method=sitk.sitkLinear)
                resampled_moving_mask = sitk.GetArrayFromImage(resampled_moving_mask)
                resampled_moving_mask[resampled_moving_mask < 0.5] = 0
                resampled_moving_mask[resampled_moving_mask > 0] = 1
                resampled_moving_mask = sitk.GetImageFromArray(resampled_moving_mask.astype('int8'))
                resampled_moving_mask.SetOrigin(resampled_moving_image.GetOrigin())
                resampled_moving_mask.SetDirection(resampled_moving_image.GetDirection())
                resampled_moving_mask.SetSpacing(resampled_moving_image.GetSpacing())

                sitk.WriteImage(resampled_moving_image, os.path.join(nifti_export_path,
                                                                     '{}_Secondary_Dicom.nii'.format(patient_id)))
                sitk.WriteImage(resampled_moving_mask, os.path.join(nifti_export_path,
                                                                    '{}_Secondary_Mask.nii'.format(patient_id)))
                sitk.WriteImage(fixed_dicom_image, os.path.join(nifti_export_path,
                                                                '{}_Primary_Dicom.nii'.format(patient_id)))
                sitk.WriteImage(fixed_dicom_mask, os.path.join(nifti_export_path,
                                                               '{}_Primary_Mask.nii'.format(patient_id)))
                if add_patient:
                    new_patient = {'PatientID': patient_id, 'MRN': MRN}
                    patient_df = patient_df.append(new_patient, ignore_index=True)
                    patient_df.to_excel(anonymized_sheet, index=0)


if __name__ == "__main__":
    pass
    # register_images_to_nifti()
