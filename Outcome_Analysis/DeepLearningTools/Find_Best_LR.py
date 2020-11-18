__author__ = 'Brian M Anderson'
# Created on 4/26/2020
from Deep_Learning.Base_Deeplearning_Code.Plot_And_Scroll_Images.Plot_Scroll_Images import plot_scroll_Image
import tensorflow as tf
from Base_Deeplearning_Code.Finding_Optimization_Parameters.LR_Finder import LearningRateFinder
from tensorflow.keras.callbacks import TensorBoard
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, return_paths
from Base_Deeplearning_Code.Models.TF_Keras_Models import my_UNet


def find_best_lr(optimizer='SGD', batch_size=16, path_desc='', bn_before_activation=True, add=''):
    base_dict = return_base_dict(optimizer=optimizer)
    min_lr = 1e-7
    max_lr = 1
    for iteration in [0]:
        for optimizer in ['Adam']:
            for atrous in [False]:
                for layer in [4, 3, 2]:
                    for max_conv_blocks in [4, 6, 8]:
                        for filters in [32]:
                            for max_filters in [128]:
                                for num_conv_blocks in [3]:
                                    for conv_lambda in [2, 1]:
                                        if layer == 1 and conv_lambda > 0:
                                            continue
                                        if num_conv_blocks + conv_lambda * (layer - 1) <= (
                                                max_conv_blocks - 2) and max_conv_blocks > 4:
                                            continue
                                        base_path, morfeus_drive = return_paths()
                                        run_data = base_dict(min_lr=min_lr, max_lr=max_lr, filters=filters, max_filters=max_filters,
                                                             layers=layer, conv_lambda=conv_lambda, num_conv_blocks=num_conv_blocks,
                                                             atrous=atrous, max_conv_blocks=max_conv_blocks)
                                        layers_dict = get_layers_dict_new(**run_data, bn_before_activation=bn_before_activation)
                                        things = ['max_conv_blocks_{}'.format(max_conv_blocks), 'layers_{}'.format(layer),'num_conv_blocks_{}'.format(num_conv_blocks),
                                                  'conv_lambda_{}'.format(conv_lambda),'filters_{}'.format(filters),
                                                  'max_filters_{}'.format(max_filters), 'Optimizer_{}'.format(optimizer)]
                                        if atrous:
                                            things = ['atrous'] + things
                                        else:
                                            things = ['conv'] + things
                                        things.append('{}_Iteration'.format(iteration))
                                        out_path = os.path.join(morfeus_drive,path_desc,'Fully_Atrous')
                                        for thing in things:
                                            out_path = os.path.join(out_path,thing)
                                        if os.path.exists(out_path):
                                            print('already done')
                                            continue
                                        os.makedirs(out_path)
                                        print(out_path)
                                        base_path, morfeus_drive, train_generator, validation_generator = return_generators(
                                            batch_size=batch_size, add=add)
                                        model = my_UNet(layers_dict=layers_dict, image_size=(None, None, None, 1),
                                                        mask_output=True).created_model
                                        k = TensorBoard(log_dir=out_path, profile_batch=0, write_graph=True)
                                        k.set_model(model)
                                        k.on_train_begin()
                                        # optimizer = tf.keras.optimizers.Adam()
                                        # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
                                        # model.compile(optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                        #               metrics=[tf.keras.metrics.SparseCategoricalAccuracy(),
                                        #                        SparseCategoricalMeanDSC(num_classes=2)])
                                        # model.fit(train_generator.data_set, epochs=5, steps_per_epoch=20,
                                        #           validation_data=validation_generator.data_set, validation_steps=5)
                                        # data = next(iter(validation_generator.data_set))
                                        # pred = model(data[0])
                                        if optimizer == 'SGD':
                                            lr_opt = tf.keras.optimizers.SGD
                                        else:
                                            lr_opt = tf.keras.optimizers.Adam
                                        LearningRateFinder(epochs=10, model=model, metrics=['sparse_categorical_accuracy'],
                                                           out_path=out_path, optimizer=lr_opt,
                                                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                                                           steps_per_epoch=len(train_generator),
                                                           train_generator=train_generator.data_set, lower_lr=min_lr, high_lr=max_lr)
                                        tf.keras.backend.clear_session()
                                        return None # repeat!


if __name__ == '__main__':
    pass
