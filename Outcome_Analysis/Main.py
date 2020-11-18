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
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Register_Images import register_images_to_nifti
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
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.CheckLiverVolume import check_liver_volume
    check_liver_volume(nifti_export_path=nifti_export_path)

'''
Ensure that all contours are within the liver contour, as sometimes they're drawn to extend past it
'''

Contour_names = ['Retro_GTV', 'Retro_GTV_Recurred', 'Liver']
write_records = False
if write_records:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Nifti_to_tfrecords import nifti_to_records
    nifti_to_records(nifti_path=nifti_export_path)


'''
Now lets split them up into 5 cross-validation groups, based on patient ID
'''
distribute_into_groups = False
if distribute_into_groups:
    from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.DistributeIntoCVGroups import distribute_into_cv, os
    records_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records'
    out_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\CrossValidation'
    description = '_No_Recurrence'
    for description in ['No_Recurrence', 'Recurrence']:
        distribute_into_cv(records_path=os.path.join(records_path, description), out_path_base=out_path,
                           description='_{}'.format(description), cv_groups=5)

'''
Now, we can finally work on the deep learning part, these will be in DeepLearningTools
'''
workondeeplearning = False
if workondeeplearning:
    from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import Data_Generator_Class
    from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
    from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
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