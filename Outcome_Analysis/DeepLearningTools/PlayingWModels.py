__author__ = 'Brian M Anderson'
# Created on 11/18/2020

from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_paths, return_generators, plot_scroll_Image
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss, cosine_loss
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import return_model
from Deep_Learning.Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR
from Deep_Learning.Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import SGDRScheduler
import tensorflow as tf
from tensorflow.keras import metrics
import os
import numpy as np

loss = CosineLoss()
# loss = tf.keras.losses.CategoricalCrossentropy()
model_key = 5
tf.random.set_seed(3141)
id = 20
base_path, morfeus_drive, train_generator, validation_generator = return_generators(evaluate=False, batch_size=32,
                                                                                    cache=True, cache_add='Playing',
                                                                                    model_key=model_key)
x, y = next(iter(validation_generator.data_set))
model_path_dir = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_2{}'.format(id)
model_path = os.path.join(model_path_dir, 'cp.ckpt')
tf_path = r'K:\Morfeus\BMAnderson\Modular_Projects\Liver_Local_Recurrence_Work\Predicting_Recurrence\Tensorflow\Test\Model_{}'.format(id)
model = return_model(model_key=model_key)
keys = {'blocks_in_dense': 1, 'dense_conv_blocks': 3, 'dense_layers': 1, 'num_dense_connections': 64, 'filters': 16,
        'growth_rate': 16, 'reduction': 1., 'dropout': 0., 'channels': 3, 'global_max': True}
if model_key > 2:
    model = model(**keys)
# if not os.path.exists(model_path_dir) or not os.listdir(model_path_dir):
    # loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(lr=5e-4)
checkpoint = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True,
                                                save_freq='epoch', save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tf_path,# profile_batch='50,100',
                                             write_graph=True)  # profile_batch='300,401',
lrate = SGDRScheduler(min_lr=1e-5, max_lr=1e-4, steps_per_epoch=len(train_generator), cycle_length=1000,
                      lr_decay=0.5, mult_factor=1, gentle_start_epochs=0, gentle_fraction=1.0)
add_lr = Add_Images_and_LR(log_dir=tf_path, add_images=False)
callbacks = [tensorboard, lrate, add_lr, checkpoint]
METRICS = [
    metrics.Precision(name='Precision'),
    metrics.Recall(name='Recall'),
    metrics.AUC(name='AUC'),
]
model.compile(optimizer, loss=loss, metrics=METRICS)
# model.train_on_batch(x, y)
# pred = model.predict(x)
# cosine_loss(y_true=y, y_pred=pred)
model.fit(train_generator.data_set, epochs=10001, steps_per_epoch=len(train_generator),
          validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
          validation_freq=5, callbacks=callbacks)

# model.load_weights(model_path)
# xxx = 1
# pred_list = []
# truth_list = []
# val_iterator = iter(validation_generator.data_set)
# for i in range(len(validation_generator)):
#     print(i)
#     x, y = next(val_iterator)
#     pred = model.predict(x)
#     output_pred = np.argmax(pred)
#     pred_list.append(output_pred)
#     truth_list.append(np.argmax(y[0].numpy()))
# print(truth_list)
# print(pred_list)