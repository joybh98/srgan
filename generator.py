#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorflow.keras.layers import BatchNormalization,Conv2D
from tensorlayer.layers import (Elementwise)

def generator():
    inputs = tf.keras.Input(shape=(32,32,3))
    n = tf.keras.layers.Conv2D(32,(3,3),strides=(1,1),padding="same",activation='relu')(inputs)
    temp = n

    # Identity Block
    for i in range(15):
        nn = tf.keras.layers.Conv2D(32,(3,3),strides=(1,1),padding="same",activation='relu')(inputs)
        nn = tf.keras.layers.BatchNormalization()        
        nn = tf.keras.layers.Conv2D(32,(3,3),strides=(1,1),padding="same",activation='relu')(inputs)
        nn = tf.keras.layers.BatchNormalization()
        # TODO: Implement elementwise layer
        #n = nn 
        break
    outputs = tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='tanh')(n)
    model = tf.keras.Model(inputs=inputs,outputs=outputs)

    return model

def compile(model):
    model.compile(optimizer='rmsprop')

genmodel = generator()
genmodel.compile()
genmodel.summary()
