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


def write_to_excel(excel_path, iterations, batch_size):
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[pd.isnull(base_df.Iteration) & ~pd.isnull(base_df.min_lr)
                                      & (base_df['Optimizer'] == 'Adam')
                                      & (base_df['run?'] == -10)
                                      ]
    indexes_for_not_run = potentially_not_run.index.values
    guess_index = 0
    rewrite = False
    for index in indexes_for_not_run:
        for iteration in iterations:
            run_df = base_df.loc[[index]]
            run_df.at[index, 'Iteration'] = iteration
            run_df.at[index, 'batch_size'] = batch_size
            model_key = run_df.loc[index, 'Model_Type']
            compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration', 'Optimizer', 'loss',
                            'batch_size', 'run?')
            if model_key > 2:
                compare_list = ('Model_Type', 'min_lr', 'max_lr', 'step_factor', 'Iteration',
                                'blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'num_dense_connections',
                                'filters', 'growth_rate', 'Dropout', 'global_max',
                                'Optimizer', 'loss', 'reduction', 'batch_size', 'run?')
            contained = is_df_within_another(data_frame=base_df, current_run_df=run_df, features_list=compare_list)
            if not contained:
                while guess_index in base_df['Model_Index'].values:
                    guess_index += 1
                run_df.at[index, 'reference'] = run_df.loc[index, 'Model_Index']
                run_df.at[index, 'Model_Index'] = guess_index
                run_df.set_index('Model_Index')
                base_df = base_df.append(run_df, ignore_index=True)
                rewrite = True
    if rewrite:
        base_df.to_excel(excel_path, index=0)


def run_2d_model(batch_size=24):
    tf.random.set_seed(3141)
    epochs = 100001
    base_path, morfeus_drive, excel_path = return_paths()
    iterations = [0, 1]
    if base_path.startswith('H'):
        write_to_excel(excel_path=excel_path, iterations=iterations, batch_size=batch_size)
    channel_keys = {3: 2, 4: 3, 5: 3, 7: 4, 8: 2, 9: 4, 10: 3, 11: 5, 12: 3}
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    potentially_not_run = base_df.loc[~pd.isnull(base_df.Iteration)
                                      & (base_df['Optimizer'] == 'Adam')
                                      & (base_df['run?'] == -10)
                                      & (base_df['Model_Type'] == 10)
                                      & pd.isnull(base_df['epoch_loss'])
                                      ]
    indexes_for_not_run = potentially_not_run.index.values
    np.random.shuffle(indexes_for_not_run)
    for index in indexes_for_not_run:
        run_df = base_df.loc[[index]]
        model_key = run_df.loc[index, 'Model_Type']
        model_index = run_df.loc[index, 'Model_Index']
        tensorboard_path = os.path.join(morfeus_drive, 'Tensorflow', 'Model_Key_{}'.format(model_key),
                                        'Model_Index_{}'.format(model_index))
        if os.path.exists(tensorboard_path):
            continue
        os.makedirs(tensorboard_path)
        features_list = ('Model_Type', 'step_factor', 'Optimizer', 'min_lr', 'max_lr', 'loss')
        if model_key > 2:
            features_list = ('Model_Type', 'step_factor', 'blocks_in_dense', 'dense_conv_blocks', 'dense_layers',
                             'num_dense_connections', 'filters', 'growth_rate', 'Optimizer', 'min_lr', 'max_lr',
                             'loss', 'Dropout', 'global_max')
        _, _, train_generator, validation_generator = return_generators(batch_size=batch_size,
                                                                        cache_add='main_{}'.format(model_key),
                                                                        cache=True, model_key=model_key)
        model_base = return_model(model_key=model_key)
        model_parameters = run_df.squeeze().to_dict()
        model_parameters['channels'] = channel_keys[model_key]
        for key in model_parameters.keys():
            if type(model_parameters[key]) is np.int64:
                model_parameters[key] = int(model_parameters[key])
            elif type(model_parameters[key]) is np.float64:
                model_parameters[key] = float(model_parameters[key])
        if model_parameters['loss'] == 'CosineLoss':
            loss = CosineLoss()
        elif model_parameters['loss'] == 'SigmoidFocal':
            loss = SigmoidFocalCrossEntropy()
        elif model_parameters['loss'] == 'CategoricalCrossEntropy':
            loss = tf.keras.losses.CategoricalCrossentropy()
        if model_parameters['Optimizer'] == 'SGD':
            opt = tf.keras.optimizers.SGD()
        elif model_parameters['Optimizer'] == 'Adam':
            opt = tf.keras.optimizers.Adam()
        if isinstance(model_base, types.FunctionType):
            model = model_base(**model_parameters)
        else:
            model = model_base
        model_path = os.path.join(base_path, 'Models', 'Model_Type_{}'.format(model_key),
                                  'Model_Index_{}'.format(model_index))

        print('Saving model to {}\ntensorboard at {}'.format(model_path, tensorboard_path))
        hparams = return_hparams(model_parameters, features_list=features_list, excluded_keys=[])
        run_model(model=model, train_generator=train_generator, validation_generator=validation_generator,
                  min_lr=model_parameters['min_lr'], max_lr=model_parameters['max_lr'], model_path=model_path,
                  tensorboard_path=tensorboard_path, trial_id=model_index, optimizer=opt, hparams=hparams,
                  step_factor=model_parameters['step_factor'], epochs=epochs, loss=loss)
    return None


if __name__ == '__main__':
    pass
