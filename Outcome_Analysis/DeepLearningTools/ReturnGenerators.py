__author__ = 'Brian M Anderson'
# Created on 11/18/2020
from Base_Deeplearning_Code.Data_Generators.TFRecord_to_Dataset_Generator import DataGeneratorClass
from Base_Deeplearning_Code.Data_Generators.Image_Processors_Module.Image_Processors_DataSet import *
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnPaths import return_paths, os


def return_generators(batch_size=5, wanted_keys={'inputs': ('image',), 'outputs': ('annotation',)},
                      cross_validation_id=0, cache=True, flip=False):
    base_path, morfeus_drive = return_paths()
    train_recurrence_path = [os.path.join(base_path, 'CV_{}'.format(cross_validation_id),
                                          'Train_Recurrence')]
    validation_recurrence_path = [os.path.join(base_path, 'CV_{}'.format(cross_validation_id),
                                               'Validation_Recurrence')]

    train_no_recurrence_path = [os.path.join(base_path, 'CV_{}'.format(cross_validation_id),
                                             'Train_No_Recurrence')]
    validation_no_recurrence_path = [os.path.join(base_path, 'CV_{}'.format(cross_validation_id),
                                                  'Validation_No_Recurrence')]
    train_recurrence_generator = DataGeneratorClass(record_paths=train_recurrence_path)
    train_generator = DataGeneratorClass(record_paths=train_no_recurrence_path)

    validation_recurrence_generator = DataGeneratorClass(record_paths=validation_recurrence_path)
    validation_generator = DataGeneratorClass(record_paths=validation_no_recurrence_path)

    train_generator.data_set.zip(train_recurrence_generator.data_set)  # Zip them together
    train_generator.total_examples += train_recurrence_generator.total_examples

    validation_generator.data_set.zip(validation_recurrence_generator.data_set)  # Zip them together
    validation_generator.total_examples += validation_recurrence_generator.total_examples

    train_processors, validation_processors = [], []
    train_processors += [
        Flip_Images(keys=('image',), flip_lr=True, flip_up_down=True, flip_3D_together=True,flip_z=True)
    ]
    train_processors += [
        Return_Outputs(wanted_keys),
        {'shuffle': len(train_generator)}]
    validation_processors += [
        Return_Outputs(wanted_keys)]
    if batch_size != 0:
        train_processors += [{'batch': batch_size}]
        validation_processors += [{'batch': 1}]
    train_processors += [{'repeat'}]
    validation_processors += [{'repeat'}]
    train_generator.compile_data_set(image_processors=train_processors, debug=False)
    validation_generator.compile_data_set(image_processors=validation_processors, debug=False)
    generators = [validation_generator]
    generators += [train_generator]
    for generator in generators: #
        data_set = iter(generator.data_set)
        for _ in range(len(generator)):
            x, y = next(data_set)
            print(x[0].shape)
    #     print(data[1][0].shape)
    # data = next(data_set)
    return base_path, morfeus_drive, train_generator, validation_generator


if __name__ == '__main__':
    pass
