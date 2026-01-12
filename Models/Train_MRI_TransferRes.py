# -*- coding: utf-8 -*-
"""
@author: elamr
This is a transfer learning implementation using ResNet50V2 with imagenet weights for Alzheimer's MRI classification.

"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import os, io

def parquet_generator(table):
    for row in table.to_pylist():
        img = row["image"]
        label = row["label"]

        if isinstance(img, dict):
            if "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"]))
            elif "data" in img and "shape" in img:
                arr = np.array(img["data"], dtype = np.uint8)
                arr = arr.reshape(img["shape"])
                img = Image.fromarray(arr)
            else:
                raise ValueError(f"Unknown image dict format: {img.keys()}")
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img))
        else:
            img = Image.fromarray(img)

        img = np.array(img, dtype = np.float32)

        if img.ndim == 2:                     
            img = np.stack([img] * 3, axis = -1)
        elif img.ndim == 3 and img.shape[-1] == 1:  
            img = np.repeat(img, 3, axis = -1)

        img = tf.keras.applications.resnet_v2.preprocess_input(img)

        yield img, np.int32(label)

def preprocess(x, y):
    x = tf.image.resize(x, (128, 128))
    return x, y

def resnet50v2(num_classes):

    EXTModel = ResNet50V2(
        include_top = False,
        weights = "imagenet",
        input_shape = (128, 128, 3)
    )

    EXTModel.trainable = True

    x = EXTModel.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation = "relu", kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(5e-4))(x)
    x = Dropout(0.1)(x)

    x = Dense(128, activation = "relu", kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(5e-4))(x)
    x = Dropout(0.1)(x)

    outputs = Dense(num_classes, activation = "softmax", kernel_initializer="glorot_uniform" )(x)

    return Model(EXTModel.input, outputs)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataloc = os.path.join(base_dir, 'data')
    table = pq.read_table(dataloc)
    pathsave = os.path.join(base_dir, "weights", "AMRI_resnet50v2.keras")

    num_samples = table.num_rows

    dataset = tf.data.Dataset.from_generator(lambda: parquet_generator(table), output_signature = (tf.TensorSpec(shape = (None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int32),),)
    dataset = dataset.shuffle(num_samples, seed = 67, reshuffle_each_iteration = False)

    train_size = int(0.8 * num_samples)
    train = dataset.take(train_size)
    test  = dataset.skip(train_size)

    train = (train.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).repeat().prefetch(tf.data.AUTOTUNE))
    test = (test.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))

    labels = pq.read_table(dataloc).column("label").to_numpy()
    num_classes = int(labels.max() + 1)

    model = resnet50v2(num_classes)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 1.5e-4, momentum = 0.9),
        loss = loss,
        metrics = ["accuracy"]
    )

    ES = EarlyStopping(
        monitor = "val_loss",
        patience = 10,
        min_delta = 0.0001,
        restore_best_weights = True
    )

    MC = ModelCheckpoint(
        filepath = pathsave,
        monitor = "val_loss",
        save_best_only = True,
        verbose = 1
    )

    steps_per_epoch = train_size // 32

    #REMINDER, DO NOT REMOVE STEPS PER EPOCH OR IT WILL NOT STOP TRAINING!!!
    model.fit(train,epochs = 64 ,steps_per_epoch = train_size // 32, validation_data = test, validation_steps = (num_samples - train_size) // 32, callbacks = [ES, MC])

if __name__ == "__main__":
    main()