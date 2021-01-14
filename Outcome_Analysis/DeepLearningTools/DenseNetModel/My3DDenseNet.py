__author__ = 'Brian M Anderson'
# Created on 12/2/2020
from tensorflow.keras import layers
import tensorflow.keras.regularizers as regularizers
from tensorflow.keras.models import Model
from tensorflow_addons.layers import GroupNormalization


def conv_block_3d(x, growth_rate, name):
    """A building block for a dense block.

    Arguments:
    x: input tensor.
    growth_rate: float, growth rate at dense layers.
    name: string, block label.

    Returns:
    Output tensor for the block.
    """
    x1 = GroupNormalization(groups=2, axis=-1, name=name + '_0_gn')(x)
    # x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = layers.Activation('selu', name=name + '_0_selu')(x1)
    x1 = layers.Conv3D(2 * growth_rate, 1, use_bias=False, name=name + '_1_conv', padding='same')(x1)
    # x1 = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = GroupNormalization(groups=2, axis=-1, name=name + '_1_gn')(x1)
    x1 = layers.Activation('selu', name=name + '_1_selu')(x1)
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
    # x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name=name + '_bn')(x)
    x = GroupNormalization(groups=2, axis=-1, name=name + '_gn')(x)
    x = layers.Activation('selu', name=name + '_selu')(x)
    x = layers.Conv3D(int(x.shape[-1] * reduction), 1,
                      use_bias=False, padding='same', name=name + '_conv')(x)
    x = layers.AveragePooling3D(strides, strides=strides, name=name + '_pool')(x)
    return x


def mydensenet(blocks_in_dense=2, dense_conv_blocks=2, dense_layers=1, num_dense_connections=256, filters=16,
               growth_rate=16, reduction=0.5, **kwargs):
    """
    :param blocks_in_dense: how many convolution blocks are in a single size layer
    :param dense_conv_blocks: how many dense blocks before a max pooling to occur
    :param dense_layers: number of dense layers
    :param num_dense_connections:
    :param filters:
    :param growth_rate:
    :param kwargs:
    :return:
    """
    blocks_in_dense = int(blocks_in_dense)
    dense_conv_blocks = int(dense_conv_blocks)
    dense_layers = int(dense_layers)
    num_dense_connections = int(num_dense_connections)
    filters = int(filters)
    growth_rate = int(growth_rate)
    reduction = float(reduction)
    input_shape = (32, 64, 64, 2)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    inputs = (img_input,)

    x = layers.Conv3D(filters, (3, 7, 7), strides=2, use_bias=False, name='conv1/conv', padding='Same')(x)
    # x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='conv1/bn')(x)
    x = GroupNormalization(groups=2, axis=-1, name='conv1/gn')(x)
    x = layers.Activation('selu', name='conv1/selu')(x)

    for i in range(dense_conv_blocks):
        x = dense_block3d(x=x, growth_rate=growth_rate, blocks=blocks_in_dense, name='conv{}'.format(i))
        x = transition_block(x=x, reduction=reduction, name='pool{}'.format(i))
    # x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5, name='bn')(x)
    x = GroupNormalization(groups=2, axis=-1, name='gn')(x)
    x = layers.Activation('selu', name='selu')(x)

    x = layers.AveragePooling3D(pool_size=(2, 2, 2), name='final_average_pooling')(x)
    x = layers.Flatten()(x)
    for i in range(dense_layers):
        x = layers.Dense(num_dense_connections, activation='selu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.Dropout(0.5)(x)
    x = layers.Dense(2, activation='softmax', name='prediction', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=(x,), name='my_3d_densenet')
    return model


if __name__ == '__main__':
    pass
