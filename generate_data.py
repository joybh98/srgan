import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os

_URL_ = 'http://image-net.org/small/valid_32x32.tar'
data_dir = tf.keras.utils.get_file(origin=_URL_,fname='lr_images',untar=True)
PATH = os.path.join(os.path.dirname(data_dir))

train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Read files and convert them to tensor
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

