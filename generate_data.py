import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os
import numpy as np

# Automatically adjust the tf.data runtime to tune the value dynamically at runtime for the efficiency issue
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Image retreive, uncomment if using for the first time
#_URL_ = 'http://image-net.org/small/valid_32x32.tar'
#data_dir = tf.keras.utils.get_file(origin=_URL_,fname='lr_images',untar=True)
#PATH = os.path.join(os.path.dirname(data_dir))

# Store directory path as variables
DIR = '/home/joy/.keras/datasets/valid_32x32/'
TRAIN_DIR = '/home/joy/.keras/datasets/train'
TEST_DIR = '/home/joy/.keras/datasets/test'

batch_size = 1000
train_ds = tf.data.Dataset.list_files("/home/joy/.keras/datasets/train/*.png")
test_ds = tf.data.Dataset.list_files("/home/joy/.keras/datasets/test/*.png")

def prepare_for_training(ds,shuffle_buffer_size=1000):
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_final_ds = prepare_for_training(train_ds)
test_final_ds = prepare_for_training(test_ds)
