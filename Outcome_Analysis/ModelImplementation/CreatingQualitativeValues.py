__author__ = 'Brian M Anderson'
# Created on 4/9/2021
from Deep_Learning.Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, \
    plot_scroll_Image
import pickle
import copy
from Deep_Learning.Base_Deeplearning_Code.Make_Single_Images.Image_Processors_Module.src.Processors.MakeTFRecordProcessors import *
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnModels import mydensenet
from Local_Recurrence_Work.Outcome_Analysis.DeepLearningTools.ReturnCosineLoss import CosineLoss
from tensorflow.keras import optimizers
import tensorflow as tf


def load_obj(path):
    if path.find('.pkl') == -1:
        path += '.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        out = {}
        return out


def save_obj(path, obj): # Save almost anything.. dictionary, list, etc.
    if path.find('.pkl') == -1:
        path += '.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.DEFAULT_PROTOCOL)
    return None



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


def visualize_gradients(image, truth):
    # baseline = tf.random.uniform(shape=(32, 64, 64, model_parameters['channels']), minval=0., maxval=1.)
    baseline = tf.ones(shape=(32, 64, 64, 3))
    m_steps = 50
    alphas = tf.linspace(start=0.0, stop=1.0,
                         num=m_steps + 1)  # Generate m_steps intervals for integral_approximation() below.
    interpolated_images = interpolate_images(baseline=baseline, image=image, alphas=alphas)
    path_gradients = compute_gradients(images=interpolated_images, target_class_idx=truth, model=model)
    ig = integral_approximation(gradients=path_gradients)
    out_gradients = tf.reduce_sum(tf.math.abs(ig), axis=-1)
    # out_gradients /= np.max(out_gradients)
    # image[..., 2:][image[..., 2:] == 0] = np.min(image)
    midline_image = np.concatenate([image[..., i] for i in range(image.shape[-1])], axis=1)
    midline_gradient = np.concatenate([out_gradients for i in range(image.shape[-1])], axis=1)
    return out_gradients


def return_model_parameters():
    model_parameters = {'Dropout': 0., 'blocks_in_dense': 2, 'dense_conv_blocks': 1, 'dense_layers': 2, 'reduction': 1,
                        'num_dense_connections': 32, 'filters': 16, 'global_max': 1, 'growth_rate': 16, 'channels': 3,
                        'model_key': 12, 'color': 'blue', 'description': 'Primary + Secondary Deform + GTV',
                        'Model_Index': 2418}
    return model_parameters


def return_model(model_parameters):
    for key in model_parameters.keys():
        if type(model_parameters[key]) is np.int64:
            model_parameters[key] = int(model_parameters[key])
        elif type(model_parameters[key]) is np.float64:
            model_parameters[key] = float(model_parameters[key])
    model_base_path = r'H:\Deeplearning_Recurrence_Work\Models\Model_Index_{}'.format(model_parameters['Model_Index'])
    # model_base_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records\Models\Model_252'
    model_path = os.path.join(model_base_path, 'cp-best.cpkt')
    # model_path = os.path.join(model_base_path, 'final_model.h5')

    model = mydensenet(**model_parameters)
    model.load_weights(model_path)
    model.compile(optimizer=optimizers.Adam(), loss=CosineLoss())
    return model


path = r'H:\CreatingQualitativeMaps\Pat'
processors = [
    CastHandle(image_handle_keys=('primary_image', 'secondary_image_deformed', 'primary_mask'),
               d_type_keys=('float', 'float', 'int32')),
    Resampler(resample_keys=('primary_image', 'secondary_image_deformed', 'primary_mask'), verbose=True,
              desired_output_spacing=(1., 1., 2.5),
              resample_interpolators=('Linear', 'Linear', 'Nearest'),
              post_process_resample_keys=('gradients',),
              post_process_original_spacing_keys=('primary_image',),
              post_process_interpolators=('Linear',)),
    Threshold_Images(image_key='primary_image', lower_bound=-200, upper_bound=200, divide=False),
    Threshold_Images(image_key='secondary_image_deformed', lower_bound=-200, upper_bound=200, divide=False),
    Normalize_to_annotation(image_key='primary_image', annotation_key='primary_mask',
                            annotation_value_list=[1, 2, 3], mirror_max=True),
    Normalize_to_annotation(image_key='secondary_image_deformed', annotation_key='primary_mask',
                            annotation_value_list=[1, 2, 3], mirror_max=True),
    Threshold_Images(image_key='primary_image', lower_bound=-15, upper_bound=10, divide=False),
    Threshold_Images(image_key='secondary_image_deformed', lower_bound=-15, upper_bound=10, divide=False),
    AddByValues(image_keys=('primary_image', 'secondary_image_deformed'),
                values=(2.5, 2.5)),
    DivideByValues(image_keys=('primary_image', 'secondary_image_deformed'),
                   values=(12.5, 12.5)),
    AddByValues(image_keys=('primary_image', 'secondary_image_deformed'),
                values=(1, 1)),
    DivideByValues(image_keys=('primary_image', 'secondary_image_deformed'),
                   values=(2.0, 2.0)),
    DistributeIntoCubes(images=32, rows=64, cols=64, resize_keys_tuple=('gradients',),
                        wanted_keys=('primary_image_original_spacing', 'spacing'))
]
make_picke = True
associations = {'Retro_GTV_Recurred': 'Retro_GTV'}
reader = DicomReaderWriter(Contour_Names=['Retro_GTV', 'Liver_BMA_Program_4'], associations=associations)
reader.walk_through_folders(path)
reader.get_images_and_mask()

deformed_handle = sitk.ReadImage(os.path.join(path, 'Deformed.mhd'))
primary_handle = reader.dicom_handle
primary_handle = sitk.Cast(primary_handle, sitk.sitkFloat32)
if make_picke:
    if deformed_handle.GetSize() != primary_handle.GetSize():
        # print('These are not the same for {}...'.format(MRN))
        deformed_handle = sitk.Resample(deformed_handle, primary_handle, sitk.AffineTransform(3), sitk.sitkLinear,
                                        -1000, deformed_handle.GetPixelID())
    deformed_handle.SetSpacing(primary_handle.GetSpacing())
    mask = sitk.GetImageFromArray(reader.mask)
    mask.SetSpacing(primary_handle.GetSpacing())
    mask.SetOrigin(primary_handle.GetOrigin())
    mask.SetDirection(primary_handle.GetDirection())
    primary = os.path.join(path, 'Primary.mhd')
    secondary = os.path.join(path, 'Secondary.mhd')

    data_dict = {'primary_image': primary_handle,
                 'secondary_image': copy.deepcopy(primary_handle),
                 'secondary_image_deformed': deformed_handle,
                 'primary_mask': mask}
    for processor in processors:
        data_dict = processor.pre_process(data_dict)

    build_keys = ('primary_image', 'secondary_image_deformed', 'primary_liver')
    next_processors = [
        ExpandDimensions(axis=-1, image_keys=('primary_image', 'primary_liver', 'secondary_image_deformed')),
        MaskKeys(key_tuple=('primary_liver', 'primary_liver'), from_values_tuple=(1, 2), to_values_tuple=(0, 1)),
        CombineKeys(image_keys=build_keys, output_key='combined')
    ]

    model = return_model(return_model_parameters())
    for key in data_dict:
        temp_dict = data_dict[key]
        for processor in next_processors:
            temp_dict = processor.pre_process(temp_dict)
        pred = model.predict(temp_dict['combined'][None, ...])
        confidence = np.squeeze(pred)[-1]
        gradients = visualize_gradients(image=temp_dict['combined'], truth=1)
        temp_dict['gradients'] = gradients.numpy()
        save_obj(os.path.join(path, key), temp_dict)
        xxx = 1
temp_dict = load_obj(os.path.join(path, 'Non_Recurrence_Cube_0.pkl'))
temp_dict['gradients_spacing'] = temp_dict['spacing']
for processor in processors[::-1]:
    temp_dict = processor.post_process(temp_dict)
temp_dict['primary_handle'] = primary_handle
post_prediction_processors = [
    DivideByValues(image_keys=('gradients',), values=(1/255,)),
    CastData(image_keys=('gradients',), dtypes=('int16',)),
    ConvertArrayToHandle(array_keys=('gradients',), out_keys=('gradients_handle',)),
    OrientHandleToAnother(moving_handle_keys=('gradients_handle',), fixed_handle_keys=('primary_handle',))
]
for processor in post_prediction_processors:
    temp_dict = processor.pre_process(temp_dict)
visual_handle = temp_dict['gradients_handle']
visual_handle.SetMetaData("ElementNumberOfChannels", '1')
sitk.WriteImage(visual_handle, os.path.join(path, 'Out_Gradient.mhd'))