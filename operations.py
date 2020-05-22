import tensorflow as tf
from tensorflow.nn import depth_to_space
from tensorflow.keras.layers import Lambda
""" Operations which are going to be used in the model frequently or more than once """
""" Pixel Shuffle """
""" Convert shape(N,C,H,W) where N: batch_size, C: channel, H: height, W: width to:
(N,C/r*r,H*r,W*r) where r is the shuffling factor. 
Goal is to convert the depth(channel) into space(height and width) """
""" 
    Subpixel convulution
    :param input_shape: tensor_shape, (N,C,H,W)
    :param scale: Upsample factor

"""


def SubPixelConv2d(input_shape,scale:int):
    # upsample using depth to space
        def refactor(input_shape):
            # operation with the r i.e shuffling factor
            dims = [input_shape[0],
                    input_shape[1] * scale,
                    input_shape[2] * scale,
                    int(input_shape[3] / (scale ** 2))]
            # Output tuple with updated values
            output_shape = tuple(dims)
            return output_shape
        
        def subpixel(x):
            return tf.nn.depth_to_space(x,scale)
        
        return Lambda(subpixel,output_shape=refactor, name='Subpixel')
