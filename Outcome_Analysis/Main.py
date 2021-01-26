__author__ = 'Brian M Anderson'
# Created on 9/28/2020

'''
All of the top part will be in PreProcessingTools
'''
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
Create Morfeus registrations between all of the images and export locally
Create_Deformation.py
'''

'''
Export the DICOM and RT for the primary and secondary images
Export_Patients.py
'''

'''
Check that the number of ROIs corresponds with outcome status
'''

check_rois = False
if check_rois:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.SanityChecks.Check_Rois_Status_Correspondance import check_rois
    dicom_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'
    check_rois(dicom_export_path=dicom_export_path, excel_file=excel_path)

'''
Check the volumes of the livers, just to make sure everything worked out correctly
'''
check_volume = False
if check_volume:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.SanityChecks.CheckLiverVolume import check_liver_volume
    dicom_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'
    check_liver_volume(dicom_export_path=dicom_export_path, excel_file=excel_path)

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
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Register_Images import register_images_to_nifti
    dicom_export_path = r'H:\Deeplearning_Recurrence_Work\Dicom_Exports'
    excel_path = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                 r'\Predicting_Recurrence\RetroAblation.xlsx'
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    register_images_to_nifti(dicom_export_path=dicom_export_path, nifti_export_path=nifti_export_path,
                             excel_path=excel_path, anonymized_sheet=anonymized_sheet)

'''
Pad the deformed images, write over as well
'''
pad_deformed_to_primary = False
if pad_deformed_to_primary:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Create_Secondary_Deformed_nifti\
        import create_secondary_deformed_nifti
    deformation_export_path = r'H:\Deeplearning_Recurrence_Work\Deformed_Exports'
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    create_secondary_deformed_nifti(deformation_export_path=deformation_export_path, anonymized_sheet=anonymized_sheet,
                                    nifti_export_path=nifti_export_path)

'''
Identify number of sites if they aren't already included
'''
identify_sites = False
if identify_sites:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Write_Sites_To_Sheet import write_sites_to_excel
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    write_sites_to_excel(nifti_export_path=nifti_export_path, excel_path=anonymized_sheet)
'''
Distribute into training/validation groups, based on MRN and number of sites
'''
distribute_train_validation = False
if distribute_train_validation:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Distribute_Train_Validation import distribute_train_validation
    anonymized_sheet = r'\\mymdafiles\di_data1\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work' \
                       r'\Predicting_Recurrence\Patient_Anonymization.xlsx'
    distribute_train_validation(nifti_export_path=nifti_export_path, excel_path=anonymized_sheet)
'''
Ensure that all contours are within the liver contour, as sometimes they're drawn to extend past it
'''

Contour_names = ['Retro_GTV', 'Retro_GTV_Recurred', 'Liver']
write_records = True
if write_records:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Nifti_to_tfrecords import nifti_to_records, os
    nifti_to_records(nifti_path=os.path.join(nifti_export_path, 'Train'))
    nifti_to_records(nifti_path=os.path.join(nifti_export_path, 'Validation'))


'''
Now lets split them up into 5 cross-validation groups, based on patient ID
'''
# distribute_into_groups = False
# if distribute_into_groups:
#     from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.DistributeIntoCVGroups import distribute_into_cv, os
#     records_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records'
#     out_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\CrossValidation'
#     description = '_No_Recurrence'
#     for description in ['No_Recurrence', 'Recurrence']:
#         distribute_into_cv(records_path=os.path.join(records_path, description), out_path_base=out_path,
#                            description='_{}'.format(description), cv_groups=5)
print("All finished here, now move on to MainDeepLearning.py!")
