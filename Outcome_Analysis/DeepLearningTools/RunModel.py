__author__ = 'Brian M Anderson'
# Created on 11/24/2020
import tensorflow as tf
from Deep_Learning.Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import SGDRScheduler
from Deep_Learning.Base_Deeplearning_Code.Callbacks.TF2_Callbacks import Add_Images_and_LR
import os
from tensorboard.plugins.hparams.keras import Callback


def run_model(model, train_generator, validation_generator, min_lr, max_lr, model_path, tensorboard_path, trial_id,
              optimizer, hparams=None, step_factor=8, epochs=120):
    checkpoint_path = os.path.join(model_path, 'cp-best.cpkt')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', verbose=1,
                                                    save_freq='epoch', save_best_only=True, save_weights_only=True)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, profile_batch=0,
                                                 write_graph=True)  # profile_batch='300,401',

    lrate = SGDRScheduler(min_lr=min_lr, max_lr=max_lr, steps_per_epoch=len(train_generator), cycle_length=step_factor,
                          lr_decay=0.90, mult_factor=2, gentle_start=False)
    add_lr = Add_Images_and_LR(log_dir=tensorboard_path, add_images=False)
    callbacks = [tensorboard, lrate, add_lr]
    if hparams is not None:
        hp_callback = Callback(tensorboard_path, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    callbacks += [checkpoint]
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_path))
    model.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train_generator.data_set, epochs=epochs, steps_per_epoch=len(train_generator),
              validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
              validation_freq=3, callbacks=callbacks)
    model.save(os.path.join(model_path, 'final_model.h5'))
    tf.keras.backend.clear_session()
    return None


if __name__ == '__main__':
    pass
