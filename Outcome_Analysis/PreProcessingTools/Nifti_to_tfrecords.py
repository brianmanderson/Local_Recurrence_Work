__author__ = 'Brian M Anderson'
# Created on 11/18/2020

from Local_Recurrence_Work.Outcome_Analysis.Make_Single_Images.Make_TFRecord_Class import write_tf_record
from Local_Recurrence_Work.Outcome_Analysis.Make_Single_Images.Image_Processors_Module.Image_Processors_TFRecord \
    import *


def path_parser(niftii_path, **kwargs):
    data_dict = {}
    primary_files = [i for i in os.listdir(niftii_path) if i.endswith('_Primary_Dicom.nii')]
    for file in primary_files:
        iteration = file.split('_')[0]
        data_dict[iteration] = {'primary_image_path': os.path.join(niftii_path, file),
                                'primary_mask_path': os.path.join(niftii_path, file.replace('Dicom', 'Mask')),
                                'secondary_image_path': os.path.join(niftii_path, file.replace('Primary', 'Secondary')),
                                'secondary_mask_path': os.path.join(niftii_path, file.replace('Primary_Dicom',
                                                                                              'Secondary_Mask')),
                                'file_name': '{}.tfrecord'.format(file.split('_')[0])
                                }
    return data_dict


def nifti_to_records(nifti_path):
    base_normalizer = [
        Add_Images_And_Annotations(nifti_path_keys=('primary_image_path', 'secondary_image_path',
                                                    'primary_mask_path', 'secondary_mask_path'),
                                   out_keys=('primary_image', 'secondary_image', 'primary_mask', 'secondary_mask'),
                                   dtypes=('float32', 'float32', 'int8', 'int8')),
        Resampler(resample_keys=('primary_image', 'secondary_image', 'primary_mask', 'secondary_mask'),
                  desired_output_spacing=(1., 1., 2.5),
                  resample_interpolators=('Linear', 'Linear', 'Nearest', 'Nearest')),
        Threshold_Images(image_key='primary_image', lower_bound=-200, upper_bound=200, divide=False),
        Threshold_Images(image_key='secondary_image', lower_bound=-200, upper_bound=200, divide=False),
        Normalize_to_annotation(image_key='primary_image', annotation_key='primary_mask',
                                annotation_value_list=[1, 2, 3], mirror_max=True),
        Normalize_to_annotation(image_key='secondary_image', annotation_key='secondary_mask',
                                annotation_value_list=[1], mirror_max=True),
        Threshold_Images(image_key='primary_image', lower_bound=-15, upper_bound=10, divide=False),
        Threshold_Images(image_key='secondary_image', lower_bound=-15, upper_bound=10, divide=False),
        AddByValues(image_keys=('primary_image', 'secondary_image'), values=(2.5, 2.5)),
        DivideByValues(image_keys=('primary_image', 'secondary_image'), values=(12.5, 12.5)),
        DistributeIntoRecurrenceCubes(images=32, rows=128, cols=128)
    ]
    write_tf_record(niftii_path=nifti_path, file_parser=path_parser, max_records=np.inf,
                    out_path=os.path.join(nifti_path, 'Records'),
                    recordwriter=RecordWriterRecurrence(out_path=os.path.join(nifti_path, 'Records'),
                                                        file_name_key='file_name', rewrite=True),
                    image_processors=base_normalizer, is_3D=True, thread_count=20, debug=False)


if __name__ == '__main__':
    pass
