import math
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import load_model
from keras.layers import Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping

from TFfunc import *


model_path = 'model_cnn.weights.h5'


def train_model(epochs: int, scaling_factor: int, resume=False):

    dataset_dir = pathlib.Path("BSR_bsds500.tgz")
    dataset_dir = str(dataset_dir.parents[0]) + '/BSR/BSDS500/data/images'
    train_dir = dataset_dir + '/train'
    test_dir = dataset_dir + '/test'

    crop_x, crop_y = 100*scaling_factor, 100*scaling_factor
    batch_size = 16

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=(crop_y, crop_x),
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode=None,
    )

    validation_data = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        image_size=(crop_y, crop_x),
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode=None,
    )

    train_data = train_data.map(normalize)
    validation_data = validation_data.map(normalize)

    size_x, size_y = 100, 100
    train_data = train_data.map(lambda x: (down_scale(x, size_x, size_y), grayscale(x)))
    validation_data = validation_data.map(lambda x: (down_scale(x, size_x, size_y), grayscale(x)))

    model = create_model(upscale_factor=scaling_factor)
    model.summary()

    if resume:
        model.load_weights(model_path)


    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MeanSquaredError())
    model.fit(train_data, epochs=epochs, callbacks=[EarlyStopping(monitor="loss", patience=10)], validation_data=validation_data, verbose=1)

    return model


def main(path,train=False, resume=False):

    scaling_factor = 4

    if train:
        model = train_model(100, scaling_factor, resume)
        model.save_weights(model_path) # h5 format
    
    else:
        model = create_model(upscale_factor=scaling_factor)
        keras.config.enable_unsafe_deserialization()
        model.load_weights(model_path)
        model.summary()

    original_img, downscaled = load_image(path, scaling_factor)
    res = upscale_image(downscaled, model)

    res.save("_test/pred/cnn.png")


def test_cnn(path):
    main(path)
