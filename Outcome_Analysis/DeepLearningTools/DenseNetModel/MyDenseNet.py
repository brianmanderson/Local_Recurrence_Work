__author__ = 'Brian M Anderson'
# Created on 11/18/2020
__author__ = 'Brian M Anderson'

# Created on 9/1/2020
from tensorflow.keras import backend
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.python.keras.utils import data_utils

BASE_WEIGTHS_PATH = ('https://storage.googleapis.com/tensorflow/'
                     'keras-applications/densenet/')
DENSENET121_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH + 'densenet121_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET121_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET169_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH + 'densenet169_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET169_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5')
DENSENET201_WEIGHT_PATH = (
        BASE_WEIGTHS_PATH + 'densenet201_weights_tf_dim_ordering_tf_kernels.h5')
DENSENET201_WEIGHT_PATH_NO_TOP = (
        BASE_WEIGTHS_PATH +
        'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5')


def dense_block(x, blocks, name):
    """A dense block.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x


def dense_block3d(x, blocks, name):
    """A dense block.

    Arguments:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    out_filters = x.shape[-1]
    for i in range(blocks):
        x = conv_block_3d(x, 32, name=name + '_block' + str(i + 1))
    x = bottle_neck(x=x, out_filters=out_filters, name=name)
    return x


def transition_block(x, reduction, name, strides=(1, 2, 2)):
    """A transition block.

    Arguments:
    x: input tensor.
    reduction: float, compression rate at transition layers.
    name: string, block label.

    Returns:
    output tensor for the block.
    """
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv2D(int(backend.int_shape(x)[-1] * reduction), 1,
                      use_bias=False, padding='same', name=name + '_conv')(x)
    x = layers.AveragePooling3D(strides, strides=strides, name=name + '_pool')(x)
    return x


def conv_block(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv', padding='same')(x1)
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def conv_block_3d(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv3D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv', padding='same')(x1)
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv3D(growth_rate, (3, 1, 1), padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def bottle_neck(x, out_filters, name):
    """ A bottle-neck for using 3D models
    :param x:
    :param out_filters:
    :param name:
    :return:
    """
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn_bottleneck')(x)
    x = layers.Activation('relu', name=name + '_bn_relu')(x)
    x = layers.Conv3D(out_filters, 1, use_bias=False, name=name + '_bn_conv', padding='same')(x)
    return x


def DenseNet(blocks, include_top=False, weights='imagenet', input_shape=(32, 128, 128, 2), include_3d=False,
             model_name='unique', classes=1000):
    """Instantiates the DenseNet architecture.

    Reference:
    - [Densely Connected Convolutional Networks](
      https://arxiv.org/abs/1608.06993) (CVPR 2017)

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.densenet.preprocess_input` for an example.

    Arguments:
    blocks: numbers of building blocks for the four dense layers.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    weights: one of `None` (random initialization),
      'imagenet' (pre-training on ImageNet),
      or the path to the weights file to be loaded.
    input_tensor: optional Keras tensor
      (i.e. output of `layers.Input()`)
      to use as image input for the model.
    input_shape: optional shape tuple, only to be specified
      if `include_top` is False (otherwise the input shape
      has to be `(224, 224, 3)` (with `'channels_last'` data format)
      or `(3, 224, 224)` (with `'channels_first'` data format).
      It should have exactly 3 inputs channels,
      and width and height should be no smaller than 32.
      E.g. `(200, 200, 3)` would be one valid value.
    pooling: optional pooling mode for feature extraction
      when `include_top` is `False`.
      - `None` means that the output of the model will be
          the 4D tensor output of the
          last convolutional block.
      - `avg` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a 2D tensor.
      - `max` means that global max pooling will
          be applied.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.
    classifier_activation: A `str` or callable. The activation function to use
      on the "top" layer. Ignored unless `include_top=True`. Set
      `classifier_activation=None` to return the logits of the "top" layer.

    Returns:
    A `keras.Model` instance.

    Raises:
    ValueError: in case of invalid argument for `weights`,
      or invalid input shape.
    ValueError: if `classifier_activation` is not `softmax` or `None` when
      using a pretrained top layer.
    """
    # if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `imagenet` '
    #                      '(pre-training on ImageNet), '
    #                      'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape

    img_input = layers.Input(shape=input_shape)
    x = img_input

    inputs = (img_input,)
    strides = (1, 2, 2)
    if include_3d:
        strides = (2, 2, 2)

    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv', padding='Same')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.MaxPooling3D(pool_size=strides, name='pool1')(x)


    x = dense_block(x, blocks[0], name='conv2')
    # if include_3d:
    #     x = dense_block3d(x=x, blocks=blocks[0], name='3d_conv2')
    x = transition_block(x, 0.5, name='pool2', strides=strides)
    x = dense_block(x, blocks[1], name='conv3')
    # if include_3d:
    #     x = dense_block3d(x=x, blocks=blocks[1], name='3d_conv3')
    x = transition_block(x, 0.5, name='pool3', strides=strides)
    x = dense_block(x, blocks[2], name='conv4')
    # if include_3d:
    #     x = dense_block3d(x=x, blocks=blocks[2], name='3d_conv4')
    x = transition_block(x, 0.5, name='pool4', strides=strides)
    x = dense_block(x, blocks[3], name='conv5')
    # if include_3d:
    #     x = dense_block3d(x=x, blocks=blocks[3], name='3d_conv5')

    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.MaxPooling3D(pool_size=strides, name='final_max_pooling')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(classes, activation='softmax', name='predictions', dtype='float32')(x)

    if blocks == [6, 12, 24, 16]:
        model_name = 'densenet121'
    elif blocks == [6, 12, 32, 32]:
        model_name = 'densenet169'
    elif blocks == [6, 12, 48, 32]:
        model_name = 'densenet201'
    model = Model(inputs=inputs, outputs=(x,), name=model_name)
    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='9d60b8095a5708f2dcce2bca79d332c7')
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='d699b8f76981ab1b30698df4c175e90b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='1ceb130c1ea1b78c3bf6114dbdfd8807')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = data_utils.get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='30ee3e1110167f948a6b9946edeeb738')
            elif blocks == [6, 12, 32, 32]:
                weights_path = data_utils.get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='b8c4d4c20dd625c148057b9ff1c1176b')
            elif blocks == [6, 12, 48, 32]:
                weights_path = data_utils.get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='c13680b51ded0fb44dff2d8f86ac8bb1')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


def MyDenseNet121(include_top=False,
                  weights='imagenet',
                  input_shape=(32, 128, 128, 2),
                  classes=2, include_3d=False):
    """Instantiates the Densenet121 architecture."""
    return DenseNet(blocks=[6, 12, 24, 16], include_top=include_top, weights=weights, input_shape=input_shape,
                    classes=classes, include_3d=include_3d)
