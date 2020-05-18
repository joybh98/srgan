import tensorflow as tf
import tensorlayer as tl
from tensorflow import keras
from tensorflow.keras.layers import Conv2D,BatchNormalization,Input,PReLU
from operations import SubPixelConv2d

def make_generator():
	
	w_init = tf.random_normal_initializer(stddev=0.02)
	g_init = tf.random_normal_initializer()

	x_in = Input(shape=(64,64,3)) # Shape : (None,64,64,3)

	print(" x_in shape ", x_in.shape)

	n = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu', \
		kernel_initializer=w_init,data_format="channels_last",input_shape=(64,64,3))(x_in) # TODO: Fix shape current shape: (?,60,60,64)
	n = PReLU()(n)
	
	temp = n
	
	print(" temp shape ", temp.shape)

	for i in range(15):
		nn = Conv2D(filters=64,kernel_size=(1,1),strides=(1,1),padding='same', \
		kernel_initializer=w_init,data_format="channels_last",input_shape=(64,64,3))(n)

		print(" inside residual block shape ", nn.shape)

		nn = BatchNormalization(gamma_initializer=g_init)(nn)
		nn = PReLU()(nn)
		nn = Conv2D(filters=64,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu', \
		kernel_initializer=w_init,data_format="channels_last",input_shape=(64,64,3))(nn)
		nn = BatchNormalization(gamma_initializer=g_init)(nn)
		nn = tf.add_n([n,nn])
		n = nn
		print(" nn inside resd block shape ", nn.shape)

	print(n.shape)

	n = Conv2D(filters=64,kernel_size=(1,1),strides=(1,1))(n)
	n = BatchNormalization()(n)
	n = tf.add_n([n,temp])
	
	print(" n resd block end shape ", n.shape)

	n = Conv2D(filters=256,kernel_size=(1,1),strides=(1,1))(n)
	# temp fix for "Layers cannot have the same name: fix it"
	s = n
	s = SubPixelConv2d(input_shape=(n.shape),scale=2)(n)
	n = PReLU()(n)
	
	print(n.shape)

	n = Conv2D(filters=256,kernel_size=(1,1),strides=(1,1))(n)
	n = SubPixelConv2d(input_shape=(n.shape), scale=2)(n)
	n = PReLU()(n)

	print(n.shape)

	# Output Layer
	nn = Conv2D(filters=3,kernel_size=(3,3),strides=(1,1),padding='same',activation='tanh')(n)
	
	print(" o/p layer ", nn.shape)
	gen = tf.keras.Model(inputs=x_in,outputs=nn)

	return gen

generator = make_generator()

# generator.summary()