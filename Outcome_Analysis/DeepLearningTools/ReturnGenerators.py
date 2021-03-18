__author__ = 'Brian M Anderson'
# Created on 11/18/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths, os
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.src.Processors.TFDataSetProcessors import *


def return_generators(batch_size=5, wanted_keys={'inputs': ('combined',), 'outputs': ('annotation',)},
                      all_training=False, cache=False, evaluate=False, model_key=0, cache_add=None,
                      build_keys=('primary_image','secondary_image_deformed', 'primary_liver'),
                      return_validation_generators=False, on_test=False, debug=False):
    """
    :param batch_size:
    :param wanted_keys:
    :param cross_validation_id:
    :param cache:
    :return: base_path, morfeus_drive, train_generator, validation_generator
    """
    if cache:
        assert cache_add is not None, 'You need to pass something to cache_add if caching'
    expand_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver', 'secondary_image')
    mask_annotations = None
    if model_key == 3:  # If it's not pretrained, just pass 2 images
        build_keys = ('primary_image', 'secondary_image_deformed')
    elif model_key == 4:
        build_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver')
        mask_annotations = [
            MaskKeys(key_tuple=('primary_liver',), from_values_tuple=(2,), to_values_tuple=(1,)),  # Only show liver
            Cast_Data(keys=('primary_liver',), dtypes=('float32',))
        ]
    elif model_key == 5:
        build_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver')  # Only show disease
        mask_annotations = [
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(1, 2), to_values_tuple=(0, 1))
        ]
    elif model_key == 7:  # 6 was flawed, go off 7 now
        expand_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver', 'disease')
        build_keys = ('primary_image', 'secondary_image_deformed', 'disease')  # Liver and disease present, primary_liver
        mask_annotations = [
            CreateDiseaseKey(),
            Cast_Data(keys=('disease',), dtypes=('float32',)),
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(2,), to_values_tuple=(1,)),
            # MaskOneBasedOnOther(guiding_keys=('primary_liver', 'primary_liver'),
            #                     changing_keys=('primary_image', 'secondary_image_deformed'), guiding_values=(0, 0),
            #                     methods=('equal_to', 'equal_to'), mask_values=(1, 0)),
        ]
    elif model_key == 8:  # If it's not pretrained, just pass 2 images
        build_keys = ('primary_image', 'secondary_image')
    elif model_key == 9:
        build_keys = ('primary_image', 'secondary_image', 'primary_liver', 'secondary_liver')
        expand_keys = (
            'primary_image', 'secondary_image_deformed', 'primary_liver', 'secondary_liver', 'secondary_image')
        mask_annotations = [
            MaskKeys(key_tuple=('primary_liver',), from_values_tuple=(2,), to_values_tuple=(1,)),  # Only show liver
            Cast_Data(keys=('secondary_liver',), dtypes=('float32',)),
        ]
    elif model_key == 10:
        build_keys = ('primary_image', 'secondary_image', 'primary_liver')  # Only show disease
        mask_annotations = [
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(1, 2), to_values_tuple=(0, 1))
        ]
    elif model_key == 11:
        expand_keys = ('primary_image', 'secondary_image', 'primary_liver', 'disease', 'secondary_liver')
        build_keys = ('primary_image', 'secondary_image', 'primary_liver', 'disease', 'secondary_liver')  # Liver and disease present
        mask_annotations = [
            CreateDiseaseKey(),
            Cast_Data(keys=('disease', 'secondary_liver'), dtypes=('float32', 'float32')),
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(2,), to_values_tuple=(1,))
        ]
    elif model_key == 12:
        build_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver')  # Only show disease
        mask_annotations = [
            AddConstantToImages(keys=('primary_image', 'secondary_image_deformed'), values=(1, 1)),
            MultiplyImagesByConstant(keys=('primary_image', 'secondary_image_deformed'), values=(0.5, 0.5)),
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(1, 2), to_values_tuple=(0, 1))
        ]
    elif model_key == 13:
        build_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver')  # Only show disease
        mask_annotations = [
            AddConstantToImages(keys=('primary_image', 'secondary_image_deformed'), values=(1, 1)),
            MultiplyImagesByConstant(keys=('primary_image', 'secondary_image_deformed'), values=(0.5, 0.5)),
            MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(1, 2), to_values_tuple=(0, 1)),
            ArgMax(annotation_keys=('annotation',), axis=-1), ExpandDimension(axis=-1, image_keys=('annotation',)),
            Cast_Data(keys=('annotation',), dtypes=('float32',))
        ]
    '''
    The keys within the dictionary are: 'primary_image', 'secondary_image', 'secondary_image_deformed'
    '''
    base_path, morfeus_drive, excel_path = return_paths()
    train_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'Recurrence')]
    validation_key = 'Validation'
    if on_test:
        validation_key = 'Test'
    validation_path = [os.path.join(base_path, validation_key, 'Records', 'Recurrence'),
                       os.path.join(base_path, validation_key, 'Records', 'No_Recurrence')]

    train_no_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'No_Recurrence')]
    if all_training:
        validation_path = None
    train_recurrence_generator = DataGeneratorClass(record_paths=train_recurrence_path, debug=debug)
    train_no_recurence_generator = DataGeneratorClass(record_paths=train_no_recurrence_path, debug=debug)

    validation_generator = None
    if validation_path is not None:
        validation_generator = DataGeneratorClass(record_paths=validation_path)
        validation_processors = []
        if mask_annotations is not None:
            validation_processors += mask_annotations
        validation_processors += [
            ExpandDimension(axis=-1, image_keys=expand_keys),
            # ArgMax(annotation_keys=('annotation',), axis=-1),
            Cast_Data(keys=('primary_liver',), dtypes=('float32',)),
        ]
        if cache:
            validation_processors += [
                {'cache': os.path.join(validation_path[0], 'cache{}'.format(cache_add))}
            ]

        validation_processors += [
            CombineKeys(image_keys=build_keys, output_key='combined'),
            Return_Outputs(wanted_keys)]

        validation_processors += [{'batch': 1}]
        validation_processors += [{'repeat'}]

        validation_generator.compile_data_set(image_processors=validation_processors, debug=debug)
        if return_validation_generators:
            return base_path, morfeus_drive, validation_generator

    train_processors_recurr = []
    if mask_annotations is not None:
        train_processors_recurr += mask_annotations
    train_processors_recurr += [
        ExpandDimension(axis=-1, image_keys=expand_keys),
        # ArgMax(annotation_keys=('annotation',), axis=-1),
        Cast_Data(keys=('primary_liver',), dtypes=('float32',)),
    ]

    if cache:
        train_processors_recurr += [
            {'cache': os.path.join(train_recurrence_path[0], 'cache{}'.format(cache_add))}
        ]
    train_processors_recurr += [
        CombineKeys(image_keys=build_keys, output_key='combined'),
        Return_Outputs(wanted_keys),
        {'shuffle': len(train_recurrence_generator)},
        {'batch': batch_size // 2},
        {'repeat'}
    ]
    train_recurrence_generator.compile_data_set(image_processors=train_processors_recurr, debug=debug)

    train_processors_non_recurr = []
    if mask_annotations is not None:
        train_processors_non_recurr += mask_annotations
    train_processors_non_recurr += [
        ExpandDimension(axis=-1, image_keys=expand_keys),
        # ArgMax(annotation_keys=('annotation',), axis=-1),
        Cast_Data(keys=('primary_liver',), dtypes=('float32',)),
    ]

    if cache:
        train_processors_non_recurr += [
            {'cache': os.path.join(train_no_recurrence_path[0], 'cache{}'.format(cache_add))}
        ]
    train_processors_non_recurr += [
        CombineKeys(image_keys=build_keys, output_key='combined'),
        # Flip_Images(keys=('combined',), flip_lr=True, flip_up_down=True, flip_3D_together=True, flip_z=True),
        Return_Outputs(wanted_keys),
        {'shuffle': len(train_no_recurence_generator)},
        {'batch': batch_size // 2},
        {'repeat'}
    ]
    train_no_recurence_generator.compile_data_set(image_processors=train_processors_non_recurr, debug=debug)
    '''
    Now, we want to provide the model with examples of both recurrence and non_recurrence each time
    '''
    train_dataset = tf.data.Dataset.zip((train_no_recurence_generator.data_set, train_recurrence_generator.data_set)).\
        map(lambda x, y: ((tf.concat((x[0][0], y[0][0]), axis=0),), (tf.concat((x[1][0], y[1][0]), axis=0),)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_generator = train_recurrence_generator
    train_generator.data_set = train_dataset
    train_generator.total_examples += train_no_recurence_generator.total_examples

    generators = [validation_generator]
    # x, y = next(iter(train_generator.data_set))
    # print(tf.reduce_min(x[0]))
    # print(tf.reduce_max(x[0]))
    # generators += [train_generator]
    if evaluate:
        for generator in generators:
            for i in range(2):
                outputs = []
                data_set = iter(generator.data_set)
                for _ in range(len(generator)):
                    x, y = next(data_set)
                    print(x[0].shape)
                    print(np.argmax(y[0].numpy()))
    #     print(data[1][0].shape)
    # data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    # base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=8, model_key=12,
    #                                                                                     all_training=False, cache=False)
    # xxx = 1
    pass
