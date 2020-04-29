import tensorflow as tf
import tensorlayer as tl
from tensorflow import keras
from tensorlayer.layers import Elementwise 

def make_gen():
	''' Init sequential model: sequential model is the most basic API provided\
		by tf that can provide most of the things that we need'''
	model = tf.keras.Sequential()
	#Input Layer: This layer will take our input(image 32x32) and pass it to the next layer
	model.add(tf.keras.layers.InputLayer(input_shape=(32,32,3), batch_size=1000,name='Input',activation='relu'))
	n = model.add(tf.keras.layers.PReLU())
	temp = n
	''' We have to store the o/p of the prelu as it has a skip connection to the\
		element wise sum (Elemewise) layer '''
	
	temp = n
	# Identity Block
	for i in range(15):
		nn = tf.keras.layers.Conv2D(kernel_size=(3,3),padding='same')(n)
		nn = tf.keras.layers.BatchNormalization()
		nn = tf.keras.layers.Conv2D(kernel_size=(3,3),padding='same')(n)
		nn = tf.keras.layers.BatchNormalization()
		nn = Elementwise(tf.add)([n,nn])
		n = nn
	#TODO: Residual block implementation
	return model