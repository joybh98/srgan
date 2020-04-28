import tensorflow as tf
from tensorflow import keras

def make_gen():
	# Init sequential model: sequential model is the most basic API provided by tf that can provide most of the things that we need
	model = tf.keras.Sequential()
	# Input Layer: This layer will take our input(image 32x32) and pass it to the next layer
	model.add(tf.keras.layers.InputLayer(input_shape=(32,32,3), batch_size=1000,name='Input'))

	return model