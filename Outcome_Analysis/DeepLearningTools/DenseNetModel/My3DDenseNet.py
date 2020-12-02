__author__ = 'Brian M Anderson'
# Created on 12/2/2020
from tensorflow.keras import layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model


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
    x1 = layers.Conv3D(2 * growth_rate, 1, use_bias=False, name=name + '_1_conv', padding='same')(x1)
    x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv3D(growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=-1, name=name + '_concat')([x, x1])
    return x


def dense_block3d(x, growth_rate, blocks, name):
    """A dense block.

    Arguments:
      x: input tensor.
      growth_rate: integer, rate at which channels grow
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block_3d(x, growth_rate=growth_rate, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name, strides=(2, 2, 2)):
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
    x = layers.Conv3D(int(x.shape[-1] * reduction), 1,
                      use_bias=False, padding='same', name=name + '_conv')(x)
    x = layers.AveragePooling3D(strides, strides=strides, name=name + '_pool')(x)
    return x


def mydensenet(blocks_in_dense=3, dense_conv_blocks=3, dense_layers=3, num_dense_connections=256, filters=16,
               growth_rate=16, **kwargs):
    input_shape = (32, 128, 128, 2)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    inputs = (img_input,)

    x = layers.Conv3D(filters, 3, strides=1, use_bias=False, name='conv1/conv', padding='Same')(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)

    for i in range(dense_conv_blocks):
        x = dense_block3d(x=x, growth_rate=growth_rate, blocks=blocks_in_dense, name='conv{}'.format(i))
        x = transition_block(x=x, reduction=0.5, name='pool{}'.format(i))
    x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.AveragePooling3D(pool_size=(2, 2, 2), name='final_average_pooling')(x)
    x = layers.Flatten()(x)
    for i in range(dense_layers):
        x = layers.Dense(num_dense_connections, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation='softmax', name='prediction', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=(x,), name='my_3d_densenet')
    return model


if __name__ == '__main__':
    pass
