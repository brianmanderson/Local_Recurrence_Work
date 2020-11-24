__author__ = 'Brian M Anderson'
# Created on 11/24/2020
import tensorflow as tf
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
from Deep_Learning.Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from tensorflow.keras.callbacks import TensorBoard
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
import os


def find_best_lr(batch_size=24, model_type='2D_Vanilla'):
    base_path, morfeus_drive, train_generator, validation_generator = return_generators(batch_size=batch_size,
                                                                                        cross_validation_id=-1,
                                                                                        cache=False)
    min_lr = 1e-10
    max_lr = 1
    for iteration in [0, 1, 2]:
        for optimizer in ['SGD']:
            things = ['Optimizer_{}'.format(optimizer)]
            things.append('{}_Iteration'.format(iteration))
            out_path = os.path.join(morfeus_drive, 'Learning_Rates', model_type)
            for thing in things:
                out_path = os.path.join(out_path,thing)
            if os.path.exists(out_path):
                print('already done')
                continue
            os.makedirs(out_path)
            print(out_path)
            model = return_model(model_type=model_type)
            k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
            k.set_model(model)
            k.on_train_begin()
            if optimizer == 'SGD':
                lr_opt = tf.keras.optimizers.SGD
            else:
                lr_opt = tf.keras.optimizers.Adam
            LearningRateFinder(epochs=10, model=model, metrics=['accuracy'],
                               out_path=out_path, optimizer=lr_opt,
                               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                               steps_per_epoch=10000//(len(train_generator) * 10),
                               train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
            tf.keras.backend.clear_session()
            return None # repeat!


if __name__ == '__main__':
    pass
