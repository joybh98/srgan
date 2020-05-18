import tensorflow as tf
from tensorflow.keras.layers import Conv2D,LeakyReLU,BatchNormalization,Dense,Input

def make_discriminator():

    # weight init
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1.,0.02)

    """ 
    Shape is a tuple of None because the discriminator takes 2 inputs
    1. Generated Input from the generator
    2. Train input which we already have
    It's better to let the shape be none so to avoid errors 
    """

    # TODO: Define channel dimension of the inputs
    n_in = Input(shape=(None,None,3))
    
    print(n_in.shape)

    # k3n64s1
    n = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', \
        kernel_initializer=w_init,data_format="channels_last",input_shape=(None,None,3))(n_in)
    n = LeakyReLU()(n)
    
    print(n.shape)
    # k3n64s2
    n = Conv2D(filters=64,kernel_size=(1,1),strides=(2,2),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n128s1
    n = Conv2D(filters=128,kernel_size=(1,1),strides=(1,1),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n128s2
    n = Conv2D(filters=128,kernel_size=(1,1),strides=(2,2),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n256s1
    n = Conv2D(filters=256,kernel_size=(1,1),strides=(1,1),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n256s2
    n = Conv2D(filters=256,kernel_size=(1,1),strides=(2,2),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n512s1
    n = Conv2D(filters=512,kernel_size=(1,1),strides=(1,1),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    # k3n512s2
    n = Conv2D(filters=512,kernel_size=(1,1),strides=(2,2),padding='same',\
        kernel_initializer=w_init,data_format="channels_last")(n)
    n = BatchNormalization(gamma_initializer=g_init)(n)
    n = LeakyReLU()(n)

    print(n.shape)
    n = Dense(units=1024,kernel_initializer=w_init)(n)
    n = LeakyReLU()(n)
    n = Dense(units=1,kernel_initializer=w_init)(n)

    print(n.shape)
    n = tf.math.sigmoid(n, name='o/p layer')

    print(n.shape)
    dis = tf.keras.Model(inputs=n_in,outputs=n)

    return dis

discriminator = make_discriminator()
# discriminator.summary()