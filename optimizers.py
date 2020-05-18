import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19

""" Loss Function """

#TODO: Implement Loss Functions in the paper

# used in content loss

mean_squared_error = tf.keras.losses.MeanSquaredError()

# used in generator loss and discriminator loss

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

""" Generator Loss """

def generator_loss(sr_out):
	return binary_cross_entropy(tf.ones_like(sr_out),sr_out)

def discriminator_loss(hr_out,sr_out):
	hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)
	sr_loss = binary_cross_entropy(tf.zeros_like(sr_out),sr_out)

	return hr_loss + sr_loss

# @tf.function decorator compiles a function into a callable tensorflow graph
def content_loss(hr, sr):
    sr = tf.keras.applications.vgg19.preprocess_input(sr)
    hr = tf.keras.applications.vgg19.preprocess_input(hr)
    sr_features = vgg(sr) / 12.75
    hr_features = vgg(hr) / 12.75
    return mean_squared_error(hr_features, sr_features)
