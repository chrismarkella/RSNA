import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob
import pydicom

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (5, 5),
                           activation='relu',
                           input_shape=(512, 512, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(10, (5, 5),
                           activation='relu',
                           input_shape=(512, 512, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()


class Model:
    """Creating a Model"""
    def __init__(self, input_size):
        self.input_size = input_size

    def add_layer(
            self,
            number_of_filters,
            filter_size,
    ):
        tf.keras.layers.Conv2D(10, )
