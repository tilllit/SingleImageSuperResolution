import math
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping



# region Functions

def normalize(image):
    return image / 255


def grayscale(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y


def down_scale(input, size_x, size_y):
    y = grayscale(input)
    return tf.image.resize(y, [size_y, size_x], method="area")


# A-func order: relu, tanh, sigmoid
def create_model(img_channels=1, upscale_factor=5):
    inputs = Input(shape=(None, None, img_channels))
    x = layers.Conv2D(64, 5, activation="tanh", padding="same")(inputs)
    x = layers.Conv2D(32, 3,  activation="tanh", padding="same")(x)
    x = layers.Conv2D(img_channels*(upscale_factor ** 2), 3,  activation="tanh", padding="same")(x)
    #outputs = MyLayer.call(x)

    #outputs = tf.nn.depth_to_space(x, upscale_factor)

    sub_layer = Lambda(lambda x: tf.nn.depth_to_space(x, upscale_factor))
    x = sub_layer(inputs=x)
    model = Model(inputs=inputs, outputs=x)
    #return Model(inputs, outputs)
    return model


def load_image(path, downfactor=5):
    #original_img = load_img(path)
    original_img = Image.open(path)
    downscaled = original_img.resize((original_img.size[0] // downfactor,
                                      original_img.size[1] // downfactor), Image.BICUBIC)
    return original_img, downscaled


def get_y_channel(image):
    ycbcr = image.convert("YCbCr")
    (y, cb, cr) = ycbcr.split()
    y = np.array(y)
    y = normalize(y.astype("float32"))
    return (y, cb, cr)


def upscale_image(img, model):
    y, cb, cr = get_y_channel(img)
    input = np.expand_dims(y, axis=0)
    out = model.predict(input)[0]
    out *= 255.0
    out = out.clip(0, 255)
    out = out.reshape((np.shape(out)[0], np.shape(out)[1]))
    new_y = Image.fromarray(np.uint8(out), mode="L")  # it will generate y channel of image
    new_cb = cb.resize(new_y.size, Image.BICUBIC)
    new_cr = cr.resize(new_y.size, Image.BICUBIC)
    res = Image.merge("YCbCr", (new_y, new_cb, new_cr)).convert(
        "RGB"
    )  # it will convert from YCbCr to RGB image
    return res

# endregion
