# -*- coding: utf-8 -*-
"""
@author: elamr
This is a transference learning implementation using ResNet50V2 with imagenet weights for Alzheimer's MRI classification.

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications.resnet import preprocess_input
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
    image = preprocess_input(image)
    return image, label


def main():

    train = train_data.map(process)
    test = test_data.map(process)

    EXTmodel = ResNet50V2(
        include_top=False,
        weights="imagenet",
        input_shape=(128, 128, 3),
        classes=4 
    )
    EXTmodel.trainable = True

    model = tf.keras.Sequential()
    model.add(EXTmodel)
    model.add(Flatten())
    model.add(Dense(32, activation='relu',  kernel_regularizer=regularizers.l1(0.0005)))
    model.add(Dense(64, activation='selu',  kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu',  kernel_regularizer=regularizers.l1(0.0005)))
    model.add(Dense(64, activation='selu',  kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu',  kernel_regularizer=regularizers.l1(0.0005)))
    model.add(Dense(64, activation='selu',  kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu',  kernel_regularizer=regularizers.l1(0.0005)))
    model.add(Dense(64, activation='selu',  kernel_regularizer=regularizers.l2(0.0005)))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax',))

    ES = EarlyStopping(
        monitor="val_loss",
        min_delta=0.01,
        patience=10,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
        start_from_epoch=0,
    )
    ModelCheckpoint(
        filepath=pathsa,
        monitor="val_loss",
        save_best_only=True,
        mode="auto",
        verbose=1,
        save_freq=14,

    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00015)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    fitted = model.fit(train, batch_size = 32, epochs = 42,validation_data = test, callbacks = [ES, ModelCheckpoint])

    model.save(pathsa)
 
if __name__ == "__main__":
    main()