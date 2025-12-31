# -*- coding: utf-8 -*-
"""
@author: elamr

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#kaggle
#import kaggle
#from kaggle.api.kaggle_api_extended import KaggleApi
#api = KaggleApi()
#api.authenticate()
#path = api.dataset_download_files("lukechugh/best-alzheimer-mri-dataset-99-accuracy", path='./ModelTraining/AMRI/data/', unzip=True)
#print("Done downloading dataset")

class BottleneckSELU(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride=1):
        super().__init__()

        self.conv = Sequential([
            Conv2D(out_channels, 1, activation='selu', strides=1, padding="same", kernel_initializer="lecun_normal"),
            Conv2D(out_channels, 3,  activation='selu', strides=stride, padding="same", kernel_initializer="lecun_normal"),
            Conv2D(out_channels, 1, activation='selu', strides=1, padding="same", kernel_initializer="lecun_normal"),
        ])

        self.shortcut = None

        if stride != 1:
            self.shortcut = Conv2D(
                out_channels, 1, strides=stride,
                kernel_initializer="lecun_normal"
            )

    def call(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.conv(x) + residual      

class InceptSELU(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
    def call(self, x):
        return self.conv(x)




def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    pathte = os.path.join(base_dir, 'data', 'Combined Dataset', 'train')
    pathtr = os.path.join(base_dir, 'data', 'Combined Dataset', 'test')
    pathsa = os.path.join(base_dir, 'weights', 'AMRI.keras')

    train_data = keras.utils.image_dataset_from_directory(
        directory=pathte,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(128, 128),
        color_mode='rgb',
        shuffle=True,
        verbose=True
    )

    test_data = keras.utils.image_dataset_from_directory(
        directory=pathtr,
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(128, 128),
        color_mode='rgb',
        shuffle=True,
        verbose=True
    )

    def process(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train = train_data.map(process).prefetch(tf.data.AUTOTUNE)
    test = test_data.map(process).prefetch(tf.data.AUTOTUNE)

    def build_model(num_classes):
        inputs = Input(shape=(128, 128, 3))

        x = Conv2D(64, 3, activation='selu', padding="same", kernel_initializer="lecun_normal")(inputs)

        x = BottleneckSELU(128, stride=2)(x)
        x = BottleneckSELU(128)(x)
        x = AlphaDropout(0.2)(x)

        x = BottleneckSELU(256, stride=2)(x)
        x = BottleneckSELU(256)(x)
        x = AlphaDropout(0.2)(x)

        x = GlobalAveragePooling2D()(x)
        outputs = Dense(num_classes, activation="softmax")(x)

        return Model(inputs, outputs)

    model = build_model(num_classes=train_data.cardinality().numpy())

    ES = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0,
    )
    MC = ModelCheckpoint(
        filepath=pathsa,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
        verbose=1,
        save_freq=14,

    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00015)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    fitted = model.fit(train, batch_size = 32, epochs = 42, validation_data = test, callbacks = [ES, MC])

    model.save(pathsa)
 
if __name__ == "__main__":
    main()