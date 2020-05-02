import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization,Conv2D,Input,PReLU,Add
from tensorflow.python.keras.models import Model
#TODO: Pixel shuffle
#TODO: Upsample

def generator(num_filters=64):

    inputs = Input(shape=(None,None,3))
    n = Conv2D(num_filters,kernel_size=9,padding='same')(inputs)
    n = PReLU(shared_axes=[1,2])(n)
    temp = n

    for i in range(16):
        nn = tf.keras.layers.Conv2D(num_filters, kernel_size=9,padding='same')(n)
        nn = tf.keras.layers.BatchNormalization()(n)
        nn = tf.keras.layers.Conv2D(num_filters, kernel_size=9,padding='same')(n)
        nn = tf.keras.layers.BatchNormalization()(n)
        Add()([n,nn])
        n = nn

    n = Conv2D(num_filters,kernel_size=9,padding='same')(n)
    n = tf.keras.layers.BatchNormalization()(n)
    n = Conv2D(num_filters,kernel_size=9,padding='same',activation='tanh')(n)
    
    return Model(inputs,n)

gen_G = generator()

gen_G.compile()
gen_G.summary()