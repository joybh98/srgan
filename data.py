#!/usr/bin/env python
# coding: utf-8

# In[30]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


# In[31]:


PATH = os.path.join(os.path.dirname('data/valid_32x32/'))
train_dir = os.path.join(PATH)
num_imgs_tr = len(os.listdir(train_dir))
num_imgs_tr


# In[32]:


image_generator = tf.keras.preprocessing.image.ImageDataGenerator()
# Loader params
BATCH_SIZE = 1000
IMG_HEIGHT = 32
IMG_WIDTH = 32 
STEPS_PER_EPOCH = np.ceil(num_imgs_tr/BATCH_SIZE)


# In[33]:


# Generate training data
train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT,IMG_WIDTH),
                                                    )


# In[ ]:





# In[ ]:





# In[ ]:




