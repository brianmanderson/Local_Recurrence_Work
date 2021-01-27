__author__ = 'Brian M Anderson'
# Created on 11/24/2020
import tensorflow as tf
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.HyperParameters import is_df_within_another
from tensorflow.keras.callbacks import TensorBoard
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
import os
import pandas as pd
import types
from tensorflow_addons.optimizers import RectifiedAdam


def return_model_and_things(model_base, out_path, iteration, excel_path):
    compare_keys = ('blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'num_dense_connections',
                    'filters', 'growth_rate', 'step_factor', 'Loss', 'Optimizer', 'reduction', 'Dropout')
    base_df = pd.read_excel(excel_path)
    for blocks_in_dense in [3, 5]:
        for dense_conv_blocks in [2, 3]:
            for dense_layers in [1, 3]:
                for num_dense_connections in [128, 256]:
                    for filters in [16]:
                        for reduction in [0.5, 1.0]:
                            for growth_rate in [16]:
                                for dropout in [0.0, 0.5]:
                                    new_run = {'blocks_in_dense': [blocks_in_dense],
                                               'dense_conv_blocks': [dense_conv_blocks],
                                               'dense_layers': [dense_layers],
                                               'num_dense_connections': [num_dense_connections],
                                               'filters': [filters], 'growth_rate': [growth_rate], 'run?': [0],
                                               'reduction': [reduction],
                                               'step_factor': [10], 'Loss': ['CosineLoss'], 'Optimizer': ['SGD'],
                                               'Model_Type': [3], 'Dropout': [dropout]}
                                    current_run_df = pd.DataFrame(new_run)
                                    contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                                     features_list=compare_keys)
                                    if not contained:
                                        base_df = pd.read_excel(excel_path)  # Check it once more with the latest version..
                                        contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                                         features_list=compare_keys)
                                    if contained:
                                        compare_df = base_df
                                        for key in compare_keys:
                                            compare_df = compare_df.loc[compare_df[key] == current_run_df[key].values[0]]
                                        model_index = compare_df.Model_Index.values[0]
                                        if 'epoch_loss' in compare_df.columns:
                                            if not pd.isnull(compare_df['epoch_loss'][compare_df.index.values[0]]):
                                                continue  # If it isn't null, it was already done
                                    else:
                                        model_index = 0
                                        while model_index in base_df['Model_Index'].values:
                                            model_index += 1
                                        current_run_df.insert(0, column='Model_Index', value=model_index)
                                        current_run_df.set_index('Model_Index')
                                        base_df = base_df.append(current_run_df)
                                        base_df.to_excel(excel_path, index=0)
                                    new_out_path = os.path.join(out_path, 'Model_Index_{}'.format(model_index),
                                                                '{}_Iteration'.format(iteration))
                                    if os.path.exists(new_out_path):
                                        continue
                                    try:
                                        model = model_base(blocks_in_dense=blocks_in_dense,
                                                           dense_conv_blocks=dense_conv_blocks,
                                                           dense_layers=dense_layers, dropout=dropout,
                                                           num_dense_connections=num_dense_connections, filters=filters,
                                                           growth_rate=growth_rate, reduction=reduction)
                                        return model, new_out_path
                                    except:
                                        os.makedirs(new_out_path)
                                        print('Failed to make model')
                                        continue

    return None, None


def find_best_lr(batch_size=24, model_key=0):
    base_path, morfeus_drive, excel_path = return_paths()
    min_lr = 1e-5
    max_lr = 10
    model_base = return_model(model_key=model_key)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = CosineLoss()
    features_list = ('Model_Type', 'Optimizer', 'step_factor')
    for iteration in [0, 1]:
        for optimizer in ['SGD']:
            out_path = os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_{}'.format(model_key))
            if not isinstance(model_base, types.FunctionType):
                model = model_base
                base_df = pd.read_excel(excel_path)
                current_run = {'Model_Type': [model_key], 'run?': [0], 'step_factor': [10], 'Loss': ['CosineLoss'],
                               'Optimizer': ['SGD']}
                current_run_df = pd.DataFrame(current_run)
                contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                 features_list=('Model_Type', 'Optimizer', 'step_factor'))
                if not contained:
                    model_index = 0
                    while model_index in base_df['Model_Index'].values:
                        model_index += 1
                    current_run_df.insert(0, column='Model_Index', value=model_index)
                    current_run_df.set_index('Model_Index')
                    base_df = base_df.append(current_run_df)
                    base_df.to_excel(excel_path, index=0)
                else:
                    for key in features_list:
                        base_df = base_df.loc[base_df[key] == current_run_df[key].values[0]]
                    model_index = base_df.Model_Index.values[0]
                out_path = os.path.join(out_path, 'Model_Index_{}'.format(model_index),
                                        '{}_Iteration'.format(iteration))
                if os.path.exists(out_path):
                    continue
            else:
                model, out_path = return_model_and_things(model_base=model_base, out_path=out_path, iteration=iteration,
                                                          excel_path=excel_path)
                if model is None:
                    continue
            os.makedirs(out_path)
            _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, model_key=model_key,
                                                                            all_training=True, cache=True,
                                                                            cache_add='LR_Finder')
            print(out_path)
            k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
            k.set_model(model)
            k.on_train_begin()
            if optimizer == 'SGD':
                lr_opt = tf.keras.optimizers.SGD
            elif optimizer == 'Adam':
                lr_opt = tf.keras.optimizers.Adam
            elif optimizer == 'RAdam':
                lr_opt = RectifiedAdam
            LearningRateFinder(epochs=10, model=model, metrics=['accuracy'],
                               out_path=out_path, optimizer=lr_opt,
                               loss=loss,
                               steps_per_epoch=1000,
                               train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
            tf.keras.backend.clear_session()
            return False # repeat!
    return True


if __name__ == '__main__':
    pass
