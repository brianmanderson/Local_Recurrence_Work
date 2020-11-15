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
register_export_to_nifti = False
nifti_export_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports'
if register_export_to_nifti:
    from .Register_Images import register_images_to_nifti
    dicom_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'

    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    register_images_to_nifti(dicom_export_path=dicom_export_path, nifti_export_path=nifti_export_path,
                             excel_path=excel_path, anonymized_sheet=anonymized_sheet)


'''
Ensure that all contours are within the liver contour, as sometimes they're drawn to extend past it
'''
ensure_contours = True
if ensure_contours:
    import SimpleITK as sitk
    import os
    masks = [os.path.join(nifti_export_path, i) for i in os.listdir(nifti_export_path) if i.endswith('_Mask.nii')]
    primary_masks = [i for i in masks if i.find('Primary') != -1]
    secondary = [i for i in masks if i not in primary_masks]
    for mask_path in primary_masks:
        mask = sitk.ReadImage(mask_path)
