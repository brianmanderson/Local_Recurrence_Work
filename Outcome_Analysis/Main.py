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
def path_parser(niftii_path, out_path, *args, **kwargs):
    data_dict = {}
    primary_files = [i for i in os.listdir(niftii_path) if i.endswith('_Primary_Dicom.nii')]
    for file in primary_files:
        iteration = file.split('_')[0]
        data_dict[iteration] = {'primary_image_path': os.path.join(niftii_path, file),
                                'primary_mask_path': os.path.join(niftii_path, file.replace('Dicom', 'Mask')),
                                'secondary_image_path': os.path.join(niftii_path, file.replace('Primary', 'Secondary')),
                                'secondary_mask_path': os.path.join(niftii_path, file.replace('Primary_Dicom',
                                                                                              'Secondary_Mask')),
                                'out_path': out_path,
                                'out_file': os.path.join(out_path, '{}.tfrecord'.format(file.split('_')[0]))}
    return data_dict

ensure_contours = True
if ensure_contours:
    import SimpleITK as sitk
    import os
    masks = [os.path.join(nifti_export_path, i) for i in os.listdir(nifti_export_path) if i.endswith('_Mask.nii')]
    primary_masks = [i for i in masks if i.find('Primary') != -1]
    secondary = [i for i in masks if i not in primary_masks]
    # for mask_path in primary_masks:
    #     mask = sitk.ReadImage(mask_path)
    from Local_Recurrence_Work.Outcome_Analysis.Make_Single_Images.Make_TFRecord_Class import write_tf_record
    from Local_Recurrence_Work.Outcome_Analysis.Make_Single_Images.Image_Processors_Module.Image_Processors_TFRecord import *
    check_backend = False
    thread_count = 1
    if check_backend:
        path = r'H:\Liver_Disease_Ablation'
        cube_size = (32, 128, 128)
        base_normalizer = [Normalize_to_annotation(annotation_value_list=[1, 2], mirror_max=True), To_Categorical(3)]
        image_processors_train = []
        image_processors_train += base_normalizer
        image_processors_train += [Cast_Data({'annotation': 'float16'}),
                                   Split_Disease_Into_Cubes(cube_size=cube_size, disease_annotation=2,
                                                            min_voxel_volume=300, max_voxels=1350000),
                                   Distribute_into_3D(max_z=cube_size[0], max_rows=cube_size[1], max_cols=cube_size[2],
                                                      min_z=cube_size[0])]

        write_tf_record(os.path.join(path, 'Train'),
                        out_path=os.path.join(path, 'Records', 'Train_{}_Records'.format(cube_size[0])),
                        image_processors=image_processors_train,
                        is_3D=True, rewrite=True, thread_count=thread_count)

    # path = r'H:\Liver_Disease_Ablation'
    path = nifti_export_path
    base_normalizer = [
        Add_Images_And_Annotations(nifti_path_keys=('primary_image_path', 'secondary_image_path',
                                                    'primary_mask_path', 'secondary_mask_path'),
                                   out_keys=('primary_image', 'secondary_image', 'primary_mask', 'secondary_mask'),
                                   dtypes=('float32', 'float32', 'int8', 'int8')),
        Threshold_Images(image_key='primary_image', lower_bound=-200, upper_bound=200, divide=False),
        Threshold_Images(image_key='secondary_image', lower_bound=-200, upper_bound=200, divide=False),
        Normalize_to_annotation(image_key='primary_image', annotation_key='primary_mask',
                                annotation_value_list=[1, 2, 3], mirror_max=True),
        Normalize_to_annotation(image_key='secondary_image', annotation_key='secondary_mask',
                                annotation_value_list=[1], mirror_max=True)
                       ]
    write_tf_record(niftii_path=path, file_parser=path_parser,
                    out_path=os.path.join(path, 'Records'),
                    image_processors=base_normalizer, is_3D=True, rewrite=False, thread_count=thread_count, debug=True)