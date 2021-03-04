__author__ = 'Brian M Anderson'
# Created on 2/1/2021
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths, os
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
from PIL import Image


def return_generators(batch_size=1, wanted_keys={'inputs': ('combined',), 'outputs': ('annotation',)},
                      all_training=False, build_keys=('primary_image','secondary_image_deformed', 'primary_liver'),
                      return_validation_generators=False):
    """
    :param batch_size:
    :param wanted_keys:
    :param cross_validation_id:
    :param cache:
    :return: base_path, morfeus_drive, train_generator, validation_generator
    """
    '''
    The keys within the dictionary are: 'primary_image', 'secondary_image', 'secondary_image_deformed'
    '''
    base_path, morfeus_drive, excel_path = return_paths()
    train_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'Recurrence')]
    validation_path = [os.path.join(base_path, 'Validation', 'Records', 'Recurrence'),
                       os.path.join(base_path, 'Validation', 'Records', 'No_Recurrence')]

    train_no_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'No_Recurrence')]
    if all_training:
        validation_path = None
    train_recurrence_generator = DataGeneratorClass(record_paths=train_recurrence_path)
    train_no_recurence_generator = DataGeneratorClass(record_paths=train_no_recurrence_path)

    validation_generator = None
    if validation_path is not None:
        validation_generator = DataGeneratorClass(record_paths=validation_path)
        validation_processors = [
            ExpandDimension(axis=-1, image_keys=build_keys),
            # ArgMax(annotation_keys=('annotation',), axis=-1),
            Cast_Data(keys=('primary_liver',), dtypes=('float32',))
        ]

        validation_processors += [
            CombineKeys(image_keys=build_keys, output_key='combined'),
            Return_Outputs(wanted_keys)]

        validation_processors += [{'batch': 1}]
        validation_processors += [{'repeat'}]

        validation_generator.compile_data_set(image_processors=validation_processors, debug=False)
        if return_validation_generators:
            return base_path, morfeus_drive, validation_generator

    train_processors_recurr = [
        ExpandDimension(axis=-1, image_keys=build_keys),
        # ArgMax(annotation_keys=('annotation',), axis=-1),
        Cast_Data(keys=('primary_liver',), dtypes=('float32',))
    ]
    train_processors_recurr += [
        CombineKeys(image_keys=build_keys, output_key='combined'),
        Return_Outputs(wanted_keys),
        {'batch': batch_size},
        {'repeat'}
    ]
    train_recurrence_generator.compile_data_set(image_processors=train_processors_recurr, debug=False)

    train_processors_non_recurr = [
        ExpandDimension(axis=-1, image_keys=build_keys),
        # ArgMax(annotation_keys=('annotation',), axis=-1),
        Cast_Data(keys=('primary_liver',), dtypes=('float32',))
    ]
    train_processors_non_recurr += [
        CombineKeys(image_keys=build_keys, output_key='combined'),
        # Flip_Images(keys=('combined',), flip_lr=True, flip_up_down=True, flip_3D_together=True, flip_z=True),
        Return_Outputs(wanted_keys),
        {'batch': batch_size},
        {'repeat'}
    ]
    train_no_recurence_generator.compile_data_set(image_processors=train_processors_non_recurr, debug=False)
    '''
    Now, we want to provide the model with examples of both recurrence and non_recurrence each time
    '''
    return base_path, morfeus_drive, train_no_recurence_generator, train_recurrence_generator, validation_generator


def print_center_images():
    out_image_path = r'H:\Deeplearning_Recurrence_Work\Image_exports\Rigid'
    build_keys = ('primary_image', 'secondary_image')
    _, _, train_no_recurence_generator, train_recurrence_generator, validation_generator = return_generators(batch_size=1,
                                                                                                             wanted_keys={'inputs': ('combined', 'file_name'), 'outputs': ('annotation',)},
                                                                                                             build_keys=build_keys)
    spot_dict = {}
    for generator in [train_no_recurence_generator, train_recurrence_generator, validation_generator]:
        print(generator)
        iter_generator = generator.data_set.as_numpy_iterator()
        for i in range(len(generator)):
            print(i)
            x, y = next(iter_generator)
            image = np.squeeze(x[0])
            path = x[1][0].decode()
            pat_id = path.split('.tfrecord')[0]
            if pat_id in spot_dict:
                pat_index = spot_dict[pat_id] + 1
            else:
                pat_index = 0
            spot_dict[pat_id] = pat_index
            image_path = os.path.join(out_image_path, '{}_{}.png'.format(pat_id, pat_index))
            if os.path.exists(image_path):
                continue
            primary = image[16, ..., 0]
            secondary = image[16, ..., 1]
            image = np.concatenate([primary, secondary], axis=1)
            image -= np.min(image)
            image /= np.max(image)
            image *= 255.
            # image = np.repeat(image[..., None], 3, axis=-1)
            im = Image.fromarray(image).convert('L')
            im.save(image_path)


if __name__ == '__main__':
    pass
