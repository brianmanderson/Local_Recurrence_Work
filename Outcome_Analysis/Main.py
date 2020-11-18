__author__ = 'Brian M Anderson'
# Created on 9/28/2020

'''
First thing we need to do is create segmentations of liver, GTV, and ablation on our exams
'''

'''
Run Create_Liver_Contours
Create_Liver_Contours.py
'''

'''
Run Create_Disease_Ablation_Contours
Create_Disease_Ablation_Contours.py
'''

'''
Run Create_ROI_Names
This is where we create the contour names 'Retro_GTV', 'Retro_Ablation', and 'Retro_GTV_Recurred'
The ROI will help train the model for outcomes
'''

'''
Export the DICOM and RT
Export_Patients.py
'''

'''
Register the images about the liver and export
Export_Registration.py
'''

'''
Load the DICOM and masks, register based on the exported registration matrix, export as nifti
'''
register_export_to_nifti = True
nifti_export_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports'
if register_export_to_nifti:
    from Local_Recurrence_Work.Outcome_Analysis.Register_Images import register_images_to_nifti
    dicom_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'

    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    register_images_to_nifti(dicom_export_path=dicom_export_path, nifti_export_path=nifti_export_path,
                             excel_path=excel_path, anonymized_sheet=anonymized_sheet)
'''
Check the volumes of the livers, just to make sure everything worked out correctly
'''
check_volume = False
if check_volume:
    import os
    import SimpleITK as sitk
    import numpy as np
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
        xxx = 1

'''
Ensure that all contours are within the liver contour, as sometimes they're drawn to extend past it
'''

Contour_names = ['Retro_GTV', 'Retro_GTV_Recurred', 'Liver']
write_records = False
if write_records:
    from Local_Recurrence_Work.Outcome_Analysis.Nifti_to_tfrecords import nifti_to_records
    nifti_to_records(nifti_path=nifti_export_path)


check_records = True
if check_records:
    from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import Data_Generator_Class
    from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
    generator_recurrence = Data_Generator_Class(record_paths=
                                                [r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Recurrence'])
    generator_nonrecurrence = Data_Generator_Class(record_paths=
                                                   [r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\No_Recurrence'])
    train_processors = [Return_Outputs(wanted_keys_dict={'inputs': ('image',), 'outputs': ('annotation',)})]
    generator_recurrence.compile_data_set(train_processors)
    generator_nonrecurrence.compile_data_set(train_processors)
    x, y = next(iter(generator_recurrence.data_set))
    xx, yy = next(iter(generator_nonrecurrence.data_set))
    xxx = 1