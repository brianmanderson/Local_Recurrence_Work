__author__ = 'Brian M Anderson'
# Created on 11/24/2020
import tensorflow as tf
from Deep_Learning.Base_Deeplearning_Code.Cyclical_Learning_Rate.clr_callback_TF2 import SGDRScheduler


def run_model(model, train_generator, validation_generator, min_lr, max_lr, model_path, tensorboard_path,
              hparams):
    checkpoint_path = os.path.join(model_path, 'cp-{epoch:04d}.cpkt')
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                 save_freq='epoch', save_best_only=True, save_weights_only=True, mode='min',
                                 verbose=1)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, profile_batch=0,
                                                 write_graph=True)  # profile_batch='300,401',
    lrate = SGDRScheduler(min_lr=min_lr, max_lr=max_lr, steps_per_epoch=len(train_generator), lr_decay=0.98)
    callbacks = [tensorboard, lrate]
    if include_images:
        add_images = Add_Images_and_LR(log_dir=tensorboard_output, validation_data=validation_generator.data_set,
                                       number_of_images=len(validation_generator), add_images=True,
                                       image_frequency=image_frequency,
                                       threshold_x=True)
        callbacks += [add_images]
    if hparams is not None:
        hp_callback = Callback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(trial_id))
        callbacks += [hp_callback]
    callbacks += [checkpoint]
    print('\n\n\n\nRunning {}\n\n\n\n'.format(tensorboard_output))
    model.compile(optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    model.fit(train_generator.data_set, epochs=epochs, steps_per_epoch=len(train_generator),
              validation_data=validation_generator.data_set, validation_steps=len(validation_generator),
              validation_freq=1, callbacks=callbacks)
    Model_val.save(os.path.join(model_path, 'final_model.h5'))
    tf.keras.backend.clear_session()
    return None


if __name__ == '__main__':
    pass
