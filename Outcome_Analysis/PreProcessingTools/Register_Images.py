__author__ = 'Brian M Anderson'
# Created on 11/11/2020
import os
import pandas as pd
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool.ReaderWriter import DicomReaderWriter, pydicom, plot_scroll_Image
import SimpleITK as sitk
import numpy as np
from Local_Recurrence_Work.Outcome_Analysis.RegisteringImages.src.RegisterImages.WithDicomReg import \
    register_images_with_dicom_reg


def return_MRN_dictionary(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Refined')
    df = df.loc[(df['Registered'] == 0) & (df['Has_Disease_Seg'] == 0)]
    MRN_list, primary_list, secondary_list, case_list = df['MRN'].values, df['PreExam'].values,\
                                                        df['Ablation_Exam'].values, df['Case'].values
    MRN_dictionary = {}
    for MRN, primary, secondary, case in zip(MRN_list, primary_list, secondary_list, case_list):
        if MRN in MRN_dictionary:
            pat_dict = MRN_dictionary[MRN]
        else:
            pat_dict = {'Primary': [], 'Secondary': [], 'Case_Number': []}
        if type(primary) is not float:
            primary = str(primary)
            if primary.startswith('CT'):
                if primary.find(' ') == -1:
                    primary = 'CT {}'.format(primary.split('CT')[-1])
        else:
            continue
        if type(secondary) is not float:
            secondary = str(secondary)
            if secondary.startswith('CT'):
                if secondary.find(' ') == -1:
                    secondary = 'CT {}'.format(secondary.split('CT')[-1])
        else:
            continue
        if not primary.startswith('CT') or not secondary.startswith('CT'):
            continue
        if primary not in pat_dict['Primary'] or secondary not in pat_dict['Secondary']:
            pat_dict['Primary'].append(primary)
            pat_dict['Secondary'].append(secondary)
            pat_dict['Case_Number'].append(case)
        MRN_dictionary[MRN] = pat_dict
    return MRN_dictionary


def register_images_to_nifti(dicom_export_path, nifti_export_path, excel_path, anonymized_sheet):
    patient_df = pd.read_excel(anonymized_sheet)
    MRN_dictionary = return_MRN_dictionary(excel_path)
    for MRN_key in MRN_dictionary.keys():
        MRN = str(MRN_key)
        while MRN[0] == '0':  # Drop the 0 from the front
            MRN = MRN[1:]
        if MRN not in os.listdir(dicom_export_path):
            print('{} not present in collection'.format(MRN))
            continue
        for primary, secondary, case_num in zip(MRN_dictionary[MRN_key]['Primary'],
                                                MRN_dictionary[MRN_key]['Secondary'],
                                                MRN_dictionary[MRN_key]['Case_Number']):
            """
            Check and see if we've done this one before
            """
            within = patient_df.loc[(patient_df['MRN'].astype('str') == MRN) & (patient_df['PreExam'] == primary) &
                                    (patient_df['PostExam'] == secondary)]
            add_patient = False
            if within.shape[0] != 0:
                patient_id = int(within['PatientID'].values[0])
                if os.path.exists(os.path.join(nifti_export_path, '{}_Primary_Mask.nii'.format(patient_id))):
                    print('Already written {}'.format(MRN))
                    continue
            else:
                add_patient = True
                patient_id = 0
                int_ids = [int(i) for i in patient_df['PatientID'].values]
                while patient_id in int_ids:
                    patient_id += 1
            """
            For each patient, load in the primary and secondary DICOM images, also liver.
            On primary, pull in the Retro_GTV, Recurred, these shouldn't be present on the secondary
            """
            assocations = {'Liver_BMA_Program_4': 'Liver_BMA_Program_4'}
            primary_reader = DicomReaderWriter(Contour_Names=['Retro_GTV', 'Retro_GTV_Recurred', 'Liver_BMA_Program_4'],
                                               associations=assocations, require_all_contours=False, arg_max=False)
            secondary_reader = DicomReaderWriter(Contour_Names=['Liver_BMA_Program_4'], associations=assocations)  # Only need the liver
            print(MRN)
            case_path = os.path.join(dicom_export_path, MRN, 'Case {}'.format(case_num))
            for root, directories, files in os.walk(case_path):
                if 'Registration_{}_to_{}'.format(primary, secondary) in directories and primary in directories and \
                        secondary in directories:
                    '''
                    Load in our registration
                    '''
                    registration_path = os.path.join(root, 'Registration_{}_to_{}'.format(primary, secondary))
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
                    primary_reader.walk_through_folders(primary_path)
                    has_liver = False
                    for roi in primary_reader.rois_in_case:
                        roi = roi.lower()
                        if roi in primary_reader.associations.keys() and \
                                primary_reader.associations[roi].lower() == 'liver_bma_program_4':
                            has_liver = True
                            break
                    if not has_liver:
                        print('No liver contours at {}'.format(primary_path))
                        continue
                    else:
                        has_liver = False
                        secondary_reader.walk_through_folders(secondary_path)
                        for roi in secondary_reader.rois_in_case:
                            roi = roi.lower()
                            if roi in secondary_reader.associations.keys() and \
                                    secondary_reader.associations[roi].lower() == 'liver_bma_program_4':
                                has_liver = True
                                break
                    if not has_liver:
                        print('No liver contours at {}'.format(secondary_path))
                        continue
                    primary_reader.get_images_and_mask()
                    secondary_reader.get_images_and_mask()
                    fixed_dicom_image = sitk.Cast(primary_reader.dicom_handle, sitk.sitkFloat32)
                    mask = primary_reader.mask
                    assert np.max(mask[..., 1] * mask[..., 2]) == 0, 'We have overlapping segmentations at {}' \
                                                                     ' for {} to {}'.format(MRN, primary, secondary)
                    fixed_dicom_mask = sitk.GetImageFromArray(np.argmax(mask, axis=-1).astype('int8'))
                    fixed_dicom_mask.SetSpacing(primary_reader.dicom_handle.GetSpacing())
                    fixed_dicom_mask.SetOrigin(primary_reader.dicom_handle.GetOrigin())
                    fixed_dicom_mask.SetDirection(primary_reader.dicom_handle.GetDirection())

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
                                                                           min_value=0, method=sitk.sitkNearestNeighbor)
                    resampled_moving_mask = sitk.GetArrayFromImage(resampled_moving_mask)

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
                        new_patient = {'PatientID': patient_id, 'MRN': MRN, 'PreExam': primary, 'PostExam': secondary}
                        patient_df = patient_df.append(new_patient, ignore_index=True)
                        patient_df.to_excel(anonymized_sheet, index=0)
            if not os.path.exists(os.path.join(nifti_export_path, '{}_Primary_Dicom.nii'.format(patient_id))):
                print('{} was never written!'.format(MRN))


if __name__ == "__main__":
    pass
    # register_images_to_nifti()
