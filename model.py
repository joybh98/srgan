import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, PReLU, BatchNormalization
from tensorflow.keras.models import Model

# B residual blocks are used in the generator
# the output of the PReLU is fed to the elementwise layer of the 1st block
# the output of each block is then fed to the next block's elemwsise layer


def identity_block(nn):
    # save the input value
    nn = n
    n = Conv2D()
    n = BatchNormalization()
    n = PReLU()
    n = Conv2D()
    n = BatchNormalization()
    # elementwise sum
    tf.math.add([n, nn])


def gen(input_shape):

    x_in = Input(shape=input_shape, name='Input')
    nn = Conv2D(filters=64, kernel_size=9, strides=1)(x_in)
    nn = PReLU()(nn)

    # residual blocks
    for i in range(3):
        identity_block(nn)

    nn = Conv2D(filters=256, kernel_size=3, strides=1)(nn)
    nn = BatchNormalization()
    nn = PReLU()

    nn = Conv2D(filters=256, kernel_size=3, strides=1)(nn)
    nn = BatchNormalization()
    nn = PReLU()

    # output layer
    output = Conv2D(filters=3, kernel_size=9, strides=1)

    make_gen = Model(inputs=x_in, outputs=output)

    return make_gen
