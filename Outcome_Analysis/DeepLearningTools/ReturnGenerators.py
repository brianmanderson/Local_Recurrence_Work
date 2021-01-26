__author__ = 'Brian M Anderson'
# Created on 11/18/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths, os
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *


def return_generators(batch_size=5, wanted_keys={'inputs': ('combined',), 'outputs': ('annotation',)},
                      all_training=False, cache=False, evaluate=False, model_key=0, cache_add=None,
                      build_keys=('primary_image','secondary_image_deformed', 'primary_liver'),
                      return_validation_generators=False):
    """
    :param batch_size:
    :param wanted_keys:
    :param cross_validation_id:
    :param cache:
    :return: base_path, morfeus_drive, train_generator, validation_generator
    """
    if cache:
        assert cache_add is not None, 'You need to pass something to cache_add if caching'
    if model_key > 2:  # If it's not pretrained, just pass 2 images
        build_keys = ('primary_image', 'secondary_image_deformed')
    '''
    The keys within the dictionary are: 'primary_image', 'secondary_image', 'secondary_image_deformed'
    '''
    base_path, morfeus_drive = return_paths()
    if not all_training:
        train_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'Recurrence')]
        validation_recurrence_path = [os.path.join(base_path, 'Validation', 'Records', 'Recurrence')]

        train_no_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'No_Recurrence')]
        validation_no_recurrence_path = [os.path.join(base_path, 'Validation', 'Records', 'No_Recurrence')]
    else:
        train_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'Recurrence'),
                                 os.path.join(base_path, 'Validation', 'Records', 'Recurrence')]
        validation_recurrence_path = None

        train_no_recurrence_path = [os.path.join(base_path, 'Train', 'Records', 'No_Recurrence'),
                                    os.path.join(base_path, 'Validation', 'Records', 'No_Recurrence')]
        validation_no_recurrence_path = None
    train_recurrence_generator = DataGeneratorClass(record_paths=train_recurrence_path)
    train_no_recurence_generator = DataGeneratorClass(record_paths=train_no_recurrence_path)

    validation_recurrence_generator, validation_no_recurrence_generator = None, None
    if validation_recurrence_path is not None:
        validation_recurrence_generator = DataGeneratorClass(record_paths=validation_recurrence_path)
        validation_no_recurrence_generator = DataGeneratorClass(record_paths=validation_no_recurrence_path)

    for train_generator, validation_generator, train_path, val_path in zip([train_recurrence_generator,
                                                                            train_no_recurence_generator],
                                                                           [validation_recurrence_generator,
                                                                            validation_no_recurrence_generator],
                                                                           [train_recurrence_path,
                                                                            train_no_recurrence_path],
                                                                           [validation_recurrence_path,
                                                                            validation_no_recurrence_path]):

        train_processors = [
            ExpandDimension(axis=-1, image_keys=build_keys),
            Cast_Data(key_type_dict={'primary_liver': 'float32'})
        ]
        # train_processors += [Normalize_Images(mean_val=0, std_val=0.5, image_key='primary_image'),
        #                      Normalize_Images(mean_val=0, std_val=0.5, image_key='secondary_image_deformed'),
        #                      Normalize_Images(mean_val=0, std_val=0.5, image_key='secondary_image_deformed')]
        validation_processors = [
            ExpandDimension(axis=-1, image_keys=build_keys),
            Cast_Data(key_type_dict={'primary_liver': 'float32'})
        ]
        if cache:
            train_processors += [
                {'cache': os.path.join(train_path[0], 'cache{}'.format(cache_add))}
            ]
            if validation_recurrence_path is not None:
                validation_processors += [
                    {'cache': os.path.join(val_path[0], 'cache{}'.format(cache_add))}
                ]
        train_processors += [
            CombineKeys(image_keys=build_keys, output_key='combined'),
            Flip_Images(keys=('combined',), flip_lr=True, flip_up_down=True, flip_3D_together=True, flip_z=True),
            Return_Outputs(wanted_keys),
            {'shuffle': len(train_generator)}
        ]
        validation_processors += [
            CombineKeys(image_keys=build_keys, output_key='combined'),
            Return_Outputs(wanted_keys)]
        if batch_size != 0:
            train_processors += [{'batch': batch_size//2}]
            validation_processors += [{'batch': 1}]
        train_processors += [{'repeat'}]
        validation_processors += [{'repeat'}]
        train_generator.compile_data_set(image_processors=train_processors, debug=True)
        if validation_no_recurrence_generator is not None:
            validation_generator.compile_data_set(image_processors=validation_processors, debug=True)
    if return_validation_generators:
        return base_path, morfeus_drive, validation_recurrence_generator, validation_no_recurrence_generator
    '''
    Now, we want to provide the model with examples of both recurrence and non_recurrence each time
    '''
    train_dataset = tf.data.Dataset.zip((train_no_recurence_generator.data_set, train_recurrence_generator.data_set)).\
        map(lambda x, y: ((tf.concat((x[0][0], y[0][0]), axis=0),), (tf.concat((x[1][0], y[1][0]), axis=0),)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_generator = train_recurrence_generator
    train_generator.data_set = train_dataset
    train_generator.total_examples += train_no_recurence_generator.total_examples
    validation_generator = None
    if validation_recurrence_generator is not None:
        validation_generator = validation_recurrence_generator
        validation_dataset = tf.data.Dataset.zip((validation_recurrence_generator.data_set,
                                                  validation_no_recurrence_generator.data_set)).map(
            lambda x, y: ((tf.concat((x[0][0], y[0][0]), axis=0),), (tf.concat((x[1][0], y[1][0]), axis=0),)),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_generator.data_set = validation_dataset.unbatch().batch(1)
        validation_generator.total_examples += validation_no_recurrence_generator.total_examples
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
    # base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=8,
    #                                                                                     all_training=False)
    pass
