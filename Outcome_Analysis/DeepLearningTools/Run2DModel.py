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
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
import pandas as pd
import tensorflow as tf
import types
import numpy as np


def run_2d_model(batch_size=24, model_type=3):
    tf.random.set_seed(3141)
    train_generator, validation_generator = None, None
    epochs = 10001
    base_path, morfeus_drive, excel_path = return_paths()

    iterations = [0]
    model_key_base = -1
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[pd.isnull(base_df.Iteration) & ~pd.isnull(base_df.min_lr)
                                      & (base_df['Optimizer'] == 'Adam') & (base_df['loss'] == 'CosineLoss')
                                      & (base_df['run?'] == -3) & (base_df['Model_Type'] == model_type)
                                      ]
    indexes_for_not_run = potentially_not_run.index.values
    np.random.shuffle(indexes_for_not_run)
    for index in indexes_for_not_run:
        run_df = base_df.loc[[index]]
        model_key = run_df.loc[index, 'Model_Type']
        compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration', 'Optimizer', 'loss', 'batch_size')
        features_list = ('Model_Type', 'step_factor', 'Optimizer', 'min_lr', 'max_lr', 'loss')
        if model_key > 2:
            compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration',
                            'blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'num_dense_connections',
                            'filters', 'growth_rate', 'Dropout',
                            'Optimizer', 'loss', 'reduction', 'batch_size')
            features_list = ('Model_Type', 'step_factor', 'blocks_in_dense', 'dense_conv_blocks', 'dense_layers',
                             'num_dense_connections', 'filters', 'growth_rate', 'Optimizer', 'min_lr', 'max_lr',
                             'loss', 'Dropout')
        for iteration in iterations:
            run_df.at[index, 'Iteration'] = iteration
            run_df.at[index, 'batch_size'] = batch_size
            contained = is_df_within_another(data_frame=base_df, current_run_df=run_df, features_list=compare_list)
            if contained:
                print("Already ran this one")
                continue
            else:
                base_df = pd.read_excel(excel_path, engine='openpyxl')
                base_df.set_index('Model_Index')
                contained = is_df_within_another(data_frame=base_df, current_run_df=run_df, features_list=compare_list)
                if contained:
                    print("Already ran this one")
                    continue
            model_index = 0
            tensorboard_path = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Key_{}'.format(model_key),
                                            'Model_Index_{}'.format(model_index))
            while model_index in base_df['Model_Index'].values or os.path.exists(tensorboard_path):
                model_index += 1
                tensorboard_path = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Key_{}'.format(model_key),
                                                'Model_Index_{}'.format(model_index))
            os.makedirs(tensorboard_path)
            run_df.at[index, 'reference'] = run_df.loc[index, 'Model_Index']
            run_df.at[index, 'Model_Index'] = model_index
            base_df = base_df.append(run_df)
            base_df.to_excel(excel_path, index=0)
            if model_key_base != model_key or train_generator is None:
                _, _, train_generator, validation_generator = return_generators(batch_size=batch_size,
                                                                                cache_add='main_{}'.format(model_key),
                                                                                cache=True, model_key=model_key)
                model_key_base = model_key
            model_base = return_model(model_key=model_key)
            model_parameters = run_df.squeeze().to_dict()
            model_parameters['channels'] = 2
            if run_df.loc[index, 'Model_Type'] > 3:
                model_parameters['channels'] = 3
            for key in model_parameters.keys():
                if type(model_parameters[key]) is np.int64:
                    model_parameters[key] = int(model_parameters[key])
                elif type(model_parameters[key]) is np.float64:
                    model_parameters[key] = float(model_parameters[key])
            loss = tf.keras.losses.BinaryCrossentropy()
            if model_parameters['loss'].startswith('CosineLoss'):
                loss = CosineLoss()
            elif model_parameters['loss'].startswith('SigmoidFocal'):
                loss = SigmoidFocalCrossEntropy()
            if model_parameters['Optimizer'] == 'SGD':
                opt = tf.keras.optimizers.SGD()
            elif model_parameters['Optimizer'] == 'Adam':
                opt = tf.keras.optimizers.Adam()
            if isinstance(model_base, types.FunctionType):
                model = model_base(**model_parameters)
            else:
                model = model_base
            model_path = os.path.join(base_path, 'Models', 'Model_Index_{}'.format(model_index))

            print('Saving model to {}\ntensorboard at {}'.format(model_path, tensorboard_path))
            hparams = return_hparams(model_parameters, features_list=features_list, excluded_keys=[])
            run_model(model=model, train_generator=train_generator, validation_generator=validation_generator,
                      min_lr=model_parameters['min_lr'], max_lr=model_parameters['max_lr'], model_path=model_path,
                      tensorboard_path=tensorboard_path, trial_id=model_index, optimizer=opt, hparams=hparams,
                      step_factor=model_parameters['step_factor'], epochs=epochs, loss=loss)
            return None


if __name__ == '__main__':
    pass
