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


def return_model_and_things(model_base, out_path, things, excel_path):
    compare_keys = ('blocks_in_dense', 'dense_conv_blocks', 'dense_layers', 'num_dense_connections',
                    'filters', 'growth_rate', 'step_factor', 'Loss', 'Optimizer')
    for blocks_in_dense in [5, 8]:
        for dense_conv_blocks in [5, 8]:
            for dense_layers in [0, 3, 5]:
                for num_dense_connections in [256]:
                    for filters in [8, 16]:
                        for growth_rate in [32]:
                            new_run = {'blocks_in_dense': [blocks_in_dense], 'dense_conv_blocks': [dense_conv_blocks],
                                       'dense_layers': [dense_layers], 'num_dense_connections': [num_dense_connections],
                                       'filters': [filters], 'growth_rate': [growth_rate], 'run?': [0],
                                       'step_factor': [10], 'Loss': ['CosineLoss'], 'Optimizer': ['SGD']}
                            current_run_df = pd.DataFrame(new_run)
                            all_list = 'blocks_in_dense_{}.dense_conv_blocks_{}.dense_layers_{}.' \
                                       'num_dense_connections{}.filters_{}.' \
                                       'growth_rate_{}'.format(blocks_in_dense, dense_conv_blocks, dense_layers,
                                                               num_dense_connections, filters, growth_rate)
                            new_out_path = os.path.join(out_path, all_list)
                            for thing in things:
                                new_out_path = os.path.join(new_out_path, thing)
                            if os.path.exists(new_out_path):
                                continue
                            base_df = pd.read_excel(excel_path)
                            contained = is_df_within_another(data_frame=base_df, current_run_df=current_run_df,
                                                             features_list=compare_keys)
                            if not contained:
                                model_index = 0
                                while model_index in base_df['Model_Index'].values:
                                    model_index += 1
                                current_run_df.insert(0, column='Model_Index', value=model_index)
                                current_run_df.set_index('Model_Index')
                                base_df = base_df.append(current_run_df)
                                base_df.to_excel(excel_path, index=0)
                            try:
                                model = model_base(blocks_in_dense=blocks_in_dense,
                                                  dense_conv_blocks=dense_conv_blocks, dense_layers=dense_layers,
                                                  num_dense_connections=num_dense_connections,filters=filters,
                                                  growth_rate=growth_rate)
                                return model, new_out_path
                            except:
                                os.makedirs(new_out_path)
                                print('Failed to make model')
                                continue

    return None, None


def find_best_lr(batch_size=24, model_key=0):
    base_path, morfeus_drive = return_paths()
    excel_path = os.path.join(morfeus_drive, 'ModelParameters.xlsx')
    min_lr = 1e-5
    max_lr = 10
    model_base = return_model(model_key=model_key)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    loss = CosineLoss()
    for iteration in [6]:
        for optimizer in ['SGD']:
            things = ['Optimizer_{}'.format(optimizer), 'CosineLoss']
            things.append('{}_Iteration'.format(iteration))
            out_path = os.path.join(morfeus_drive, 'Learning_Rates', 'Model_Key_{}'.format(model_key))
            if not isinstance(model_base, types.FunctionType):
                model = model_base
                for thing in things:
                    out_path = os.path.join(out_path, thing)
                if os.path.exists(out_path):
                    print('already done')
                    continue
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
                model, out_path = return_model_and_things(model_base=model_base, out_path=out_path, things=things,
                                                          excel_path=excel_path)
                if model is None:
                    continue
            os.makedirs(out_path)
            _, _, train_generator, validation_generator = return_generators(batch_size=batch_size, model_key=model_key,
                                                                            cross_validation_id=-1, cache=True)
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
            return None # repeat!


if __name__ == '__main__':
    pass
