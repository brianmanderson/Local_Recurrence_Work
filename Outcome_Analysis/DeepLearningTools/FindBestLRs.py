__author__ = 'Brian M Anderson'
# Created on 11/24/2020
import tensorflow as tf
from tensorflow.keras import metrics
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import is_df_within_another
from tensorflow.keras.callbacks import TensorBoard
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
import os
import numpy as np
import pandas as pd
import types
from tensorflow_addons.optimizers import RectifiedAdam


def create_excel_values(excel_path):
    compare_keys = ('blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'num_dense_connections',
                    'filters', 'growth_rate', 'step_factor', 'loss', 'Optimizer', 'reduction', 'Dropout', 'global_max')
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    global_max = 1
    rewrite = False
    guess_index = 0
    for model_type in [12]:
        for dropout in [0]:
            for loss in ['CosineLoss', 'CategoricalCrossEntropy']:
                for blocks_in_dense in [1, 2]:
                    for dense_conv_blocks in [1, 2]:
                        for dense_layers in [0, 1]:
                            for reduction in [1]:
                                for num_dense_connections in [256]:
                                    if dense_layers == 0 and num_dense_connections > 128:
                                        continue
                                    for filters in [16]:
                                        for growth_rate in [16]:
                                            new_run = {'blocks_in_dense': [blocks_in_dense],
                                                       'global_max': [global_max],
                                                       'dense_conv_blocks': [dense_conv_blocks],
                                                       'dense_layers': [dense_layers],
                                                       'num_dense_connections': [num_dense_connections],
                                                       'filters': [filters], 'growth_rate': [growth_rate], 'run?': [-10],
                                                       'reduction': [reduction],
                                                       'step_factor': [5000], 'loss': [loss],
                                                       'Optimizer': ['Adam'],
                                                       'Model_Type': [model_type], 'Dropout': [dropout]}
                                            current_run_df = pd.DataFrame(new_run)
                                            contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                                             features_list=compare_keys)
                                            if not contained:
                                                rewrite = True
                                                while guess_index in base_df['Model_Index'].values:
                                                    guess_index += 1
                                                current_run_df.insert(0, column='Model_Index', value=guess_index)
                                                current_run_df.set_index('Model_Index')
                                                base_df = base_df.append(current_run_df)
    if rewrite:
        base_df.to_excel(excel_path, index=0)
    return None


def return_model_parameters(out_path, excel_path, iteration):
    base_df = pd.read_excel(excel_path, engine='openpyxl')
    base_df.set_index('Model_Index')
    channel_keys = {3: 2, 4: 3, 5: 3, 7: 4, 8: 2, 9: 4, 10: 3, 11: 5, 12: 3}
    potentially_not_run = base_df.loc[pd.isnull(base_df.Iteration) & pd.isnull(base_df.min_lr)
                                      & (base_df['run?'] == -10)
                                      ]
    indexes_for_not_run = potentially_not_run.index.values
    for index in indexes_for_not_run:
        run_df = base_df.loc[[index]]
        model_parameters = run_df.squeeze().to_dict()
        model_key = run_df.loc[index, 'Model_Type']
        model_index = run_df.loc[index, 'Model_Index']
        model_out_path = os.path.join(out_path, 'Model_Key_{}'.format(model_key), 'Model_Index_{}'.format(model_index),
                                      '{}_Iteration'.format(iteration))
        if os.path.exists(model_out_path):
            continue
        os.makedirs(model_out_path)
        model_parameters['channels'] = channel_keys[model_key]
        for key in model_parameters.keys():
            if type(model_parameters[key]) is np.int64:
                model_parameters[key] = int(model_parameters[key])
            elif type(model_parameters[key]) is np.float64:
                model_parameters[key] = float(model_parameters[key])
        return model_parameters, model_out_path
    return None, None


def find_best_lr(batch_size=24):
    tf.random.set_seed(3141)
    base_path, morfeus_drive, excel_path = return_paths()

    # if base_path.startswith('H'):  # Only run this locally
    #     create_excel_values(excel_path=excel_path)
    for iteration in [0]:
        out_path = os.path.join(morfeus_drive, 'Learning_Rates')
        model_parameters, out_path = return_model_parameters(out_path=out_path, excel_path=excel_path,
                                                             iteration=iteration)
        if model_parameters is None:
            continue
        model_key = model_parameters['Model_Type']
        optimizer = model_parameters['Optimizer']
        model_base = return_model(model_key=model_key)
        model = model_base(**model_parameters)
        if model_parameters['loss'] == 'CosineLoss':
            loss = CosineLoss()
            min_lr = 1e-6
            max_lr = 1e-1
        elif model_parameters['loss'] == 'CategoricalCrossEntropy':
            loss = tf.keras.losses.CategoricalCrossentropy()
            min_lr = 1e-10
            max_lr = 1e-3
        _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, model_key=model_key,
                                                                        all_training=True, cache=True,
                                                                        cache_add='LR_Finder_{}'.format(model_key))
        print(out_path)
        k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
        k.set_model(model)
        k.on_train_begin()
        lr_opt = tf.keras.optimizers.Adam
        if optimizer == 'SGD':
            lr_opt = tf.keras.optimizers.SGD
        elif optimizer == 'Adam':
            lr_opt = tf.keras.optimizers.Adam
        elif optimizer == 'RAdam':
            lr_opt = RectifiedAdam
        METRICS = [
            metrics.TruePositives(name='TruePositive'),
            metrics.FalsePositives(name='FalsePositive'),
            metrics.TrueNegatives(name='TrueNegative'),
            metrics.FalseNegatives(name='FalseNegative'),
            metrics.CategoricalAccuracy(name='Accuracy'),
            metrics.Precision(name='Precision'),
            metrics.Recall(name='Recall'),
            metrics.AUC(name='AUC'),
        ]
        LearningRateFinder(epochs=10, model=model, metrics=METRICS,
                           out_path=out_path, optimizer=lr_opt,
                           loss=loss,
                           steps_per_epoch=1000,
                           train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
        tf.keras.backend.clear_session()
        return False # repeat!
    return True


if __name__ == '__main__':
    pass
