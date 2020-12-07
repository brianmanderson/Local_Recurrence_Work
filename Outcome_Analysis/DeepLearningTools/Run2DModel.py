__author__ = 'Brian M Anderson'
# Created on 11/28/2020
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnHparameters import return_list_of_models
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import \
    is_df_within_another, return_hparams
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.RunModel import run_model
import os
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
import pandas as pd
import tensorflow as tf
import types
from tensorflow_addons.optimizers import RectifiedAdam
import numpy as np


def run_2d_model(batch_size=24, model_key=0):
    epochs = 3901
    model_dictionary = return_list_of_models(model_key=model_key)
    list_of_models = np.asarray(model_dictionary[model_key])  # A list of models to attempt to run
    perm = np.arange(len(list_of_models))
    np.random.shuffle(perm)
    list_of_models = list_of_models[perm]  # Shuffle the list of models up
    base_path, morfeus_drive = return_paths()

    excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
    compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration', 'cv_id', 'Optimizer', 'Loss')
    features_list = ('Model_Type', 'step_factor', 'Optimizer', 'min_lr', 'max_lr', 'Loss')
    if model_key == 3:
        compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration', 'cv_id', 'blocks_in_dense',
                        'dense_conv_blocks', 'dense_layers', 'num_dense_connections', 'filters', 'growth_rate',
                        'Optimizer', 'Loss')
        features_list = ('Model_Type', 'step_factor', 'blocks_in_dense', 'dense_conv_blocks', 'dense_layers',
                         'num_dense_connections', 'filters', 'growth_rate', 'Optimizer', 'min_lr', 'max_lr', 'Loss')
    model_base = return_model(model_key=model_key)
    for cv_id in range(1):
        _, _, train_generator, validation_generator = return_generators(batch_size=batch_size,
                                                                        cross_validation_id=cv_id,
                                                                        cache=True, model_key=model_key)
        for iteration in range(5):
            for model_parameters in list_of_models:
                if isinstance(model_base, types.FunctionType):
                    model = model_base(**model_parameters)
                else:
                    model = model_base
                base_df = pd.read_excel(excel_path)
                base_df.set_index('Model_Index')
                model_parameters['Iteration'] = iteration
                if model_parameters['Optimizer'] == 'SGD':
                    opt = tf.keras.optimizers.SGD()
                elif model_parameters['Optimizer'] == 'Adam':
                    opt = tf.keras.optimizers.Adam()
                elif model_parameters['Optimizer'] == 'RAdam':
                    opt = RectifiedAdam()
                if model_parameters['Loss'] == 'CCE':
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
                elif model_parameters['Loss'] == 'CosineLoss':
                    loss = CosineLoss()
                model_parameters['cv_id'] = cv_id
                current_run_df = pd.DataFrame(model_parameters, index=[0])
                contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                 features_list=compare_list)
                if contained:
                    print("Already ran this one")
                    continue
                current_model_indexes = base_df['Model_Index']
                model_index = 0
                while model_index in current_model_indexes:
                    model_index += 1
                model_path = os.path.join(base_path, 'Models', 'Model_Index_{}'.format(model_index))
                tensorboard_path = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Key_{}'.format(model_key),
                                                'Model_Index_{}'.format(model_index))
                print('Saving model to {}\ntensorboard at {}'.format(model_path, tensorboard_path))
                current_run_df.insert(0, column='Model_Index', value=model_index)
                hparams = return_hparams(model_parameters, features_list=features_list, excluded_keys=[])
                current_run_df.set_index('Model_Index')
                base_df = base_df.append(current_run_df)
                base_df.to_excel(excel_path, index=0)
                run_model(model=model, train_generator=train_generator, validation_generator=validation_generator,
                          min_lr=model_parameters['min_lr'], max_lr=model_parameters['max_lr'], model_path=model_path,
                          tensorboard_path=tensorboard_path, trial_id=model_index, optimizer=opt, hparams=hparams,
                          step_factor=model_parameters['step_factor'], epochs=epochs, loss=loss)
                return None


if __name__ == '__main__':
    pass
