__author__ = 'Brian M Anderson'
# Created on 3/8/2021
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import mydensenet
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics, optimizers
import numpy as np
import os
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnGenerators import return_generators, plot_scroll_Image
import tensorflow_hub as hub
import tensorflow as tf
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss


def return_model_parameters():
    model_parameters = {'Dropout': 0., 'blocks_in_dense': 1, 'dense_conv_blocks': 1, 'dense_layers': 2, 'reduction': 1,
                        'num_dense_connections': 256, 'filters': 16, 'global_max': 1, 'growth_rate': 16, 'channels': 5,
                        'model_key': 11, 'color': 'green', 'description': 'Primary + Secondary Rigid + GTV + Liver',
                        'Model_Index': 1685}
    model_parameters = {'Dropout': 0., 'blocks_in_dense': 1, 'dense_conv_blocks': 1, 'dense_layers': 2, 'reduction': 1,
                        'num_dense_connections': 256, 'filters': 16, 'global_max': 1, 'growth_rate': 16, 'channels': 3,
                        'model_key': 5, 'color': 'b', 'description': 'Primary + Secondary Deform + GTV',
                        'Model_Index': 1431}
    model_parameters = {'Dropout': 0., 'blocks_in_dense': 1, 'dense_conv_blocks': 1, 'dense_layers': 2, 'reduction': 1,
                        'num_dense_connections': 256, 'filters': 16, 'global_max': 1, 'growth_rate': 16, 'channels': 3,
                        'model_key': 5, 'color': 'b', 'description': 'Primary + Secondary Deform + GTV',
                        'Model_Index': 1749}
    return model_parameters


def return_model(model_parameters):
    for key in model_parameters.keys():
        if type(model_parameters[key]) is np.int64:
            model_parameters[key] = int(model_parameters[key])
        elif type(model_parameters[key]) is np.float64:
            model_parameters[key] = float(model_parameters[key])
    model_base_path = r'H:\Deeplearning_Recurrence_Work\Models\Model_Index_{}'.format(model_parameters['Model_Index'])
    # model_base_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_252'
    model_path = os.path.join(model_base_path, 'cp.cpkt')
    # model_path = os.path.join(model_base_path, 'final_model.h5')

    model = mydensenet(**model_parameters)
    model.load_weights(model_path)
    model.compile(optimizer=optimizers.Adam(), loss=CosineLoss())
    return model


def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = logits[0][:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def view_gradients():
    model_parameters = return_model_parameters()
    model = return_model(model_parameters=model_parameters)
    _, _, val_generator = return_generators(batch_size=1, return_validation_generators=True,
                                            wanted_keys={'inputs': ('combined', 'file_name'),
                                                         'outputs': ('annotation',)},
                                            model_key=model_parameters['model_key'], on_test=False)
    iterator = iter(val_generator.data_set)
    i = -1
    for _ in range(len(val_generator)):
        i += 1
        print(i)
        x, y = next(iterator)
        truth = tf.argmax(tf.squeeze(y)).numpy()
        if truth == 0:
            continue
        pred = model.predict((x[0],))
        image = tf.squeeze(x[0]).numpy()
        file_path = x[1][0]
        print(file_path)
        # baseline = tf.random.uniform(shape=(32, 64, 64, model_parameters['channels']), minval=0., maxval=1.)
        baseline = tf.ones(shape=(32, 64, 64, model_parameters['channels']))
        m_steps = 50
        alphas = tf.linspace(start=0.0, stop=1.0,
                             num=m_steps + 1)  # Generate m_steps intervals for integral_approximation() below.
        interpolated_images = interpolate_images(baseline=baseline, image=image, alphas=alphas)
        path_gradients = compute_gradients(images=interpolated_images, target_class_idx=truth, model=model)
        ig = integral_approximation(gradients=path_gradients)
        out_gradients = tf.reduce_sum(tf.math.abs(ig), axis=-1)
        out_gradients /= np.max(out_gradients)
        image[..., 2:][image[..., 2:] == 0] = np.min(image)
        midline_image = np.concatenate([image[..., i] for i in range(image.shape[-1])], axis=1)
        midline_gradient = np.concatenate([out_gradients for i in range(image.shape[-1])], axis=1)
        xxx = 1
        # plot_scroll_Image(img=image, dose=out_gradients)



if __name__ == '__main__':
    view_gradients()
