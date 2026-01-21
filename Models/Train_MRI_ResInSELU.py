# -*- coding: utf-8 -*-
"""
@author: elamr

This is a custom ResNet-Inception architecture using SELU activations for Alzheimer's MRI classification.
This implementation is not for production and is an exploration of self normalizing neural networks in residual and inception architectures.

@dataset{alzheimer_mri_dataset,
  author = {Falah.G.Salieh},
  title = {Alzheimer MRI Dataset},
  year = {2023},
  publisher = {Hugging Face},
  version = {1.0},
  url = {https://huggingface.co/datasets/Falah/Alzheimer_MRI}
}

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import os
import io

#ALL ARC COMPONENTS AND PROBLEMS ARE HERE IN THE TWO CLASSES, ALL LAYERS NEED TO BE SELU COMPATIBLE
class ResSELU(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride = 1, dropout_rate = 0.05, alpha_init = 0.5):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2D(
            out_channels, 3, padding = "same",
            kernel_initializer = "lecun_normal", use_bias = False
        )
        self.conv2 = Conv2D(
            out_channels, 3, strides = stride, padding = "same",
            kernel_initializer = "lecun_normal", use_bias = False
        )

        self.dropout = AlphaDropout(dropout_rate)
        self.shortcut = None

        self.alpha_logit = self.add_weight(
            name = "alpha_logit",
            shape = (),
            initializer = tf.keras.initializers.Constant(
                tf.math.log(alpha_init / (1.0 - alpha_init))
            ),
            trainable = True
        )

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.out_channels:
            self.shortcut = Conv2D(
                self.out_channels, 1, strides = self.stride,
                padding = "same", kernel_initializer = "lecun_normal", use_bias = False
            )
        super().build(input_shape)

    def call(self, x, training = None):
        y = tf.nn.selu(x)
        y = self.conv1(y)
        y = tf.nn.selu(y)
        y = self.conv2(y)
        y = self.dropout(y, training = training)

        shortcut = x if self.shortcut is None else self.shortcut(x)

        alpha = tf.sigmoid(self.alpha_logit)
        return alpha * shortcut + (1.0 - alpha) * y

class SELUInception(tf.keras.layers.Layer):
    def __init__(self, channels, gate_scale = 0.1):
        super().__init__()
        self.channels = channels
        self.gate_scale = gate_scale
        self.inv_sqrt2 = tf.constant(1.0 / tf.sqrt(2.0), tf.float32)
        self.inv_sqrt4 = tf.constant(1.0 / tf.sqrt(4.0), tf.float32)

        self.b1 = Conv2D(channels, 1, padding = "same",
                         kernel_initializer = "lecun_normal", use_bias = True)
        self.b3 = Conv2D(channels, 3, padding = "same",
                         kernel_initializer = "lecun_normal", use_bias = True)
        self.b5 = Conv2D(channels, 5, padding = "same",
                         kernel_initializer = "lecun_normal", use_bias = True)

        self.pool = MaxPooling2D(3, strides = 1, padding = "same")
        self.bp = Conv2D(channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias=True)

        self.gate = Conv2D(channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)

        self.proj = None

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.proj = Conv2D(
                self.channels, 1, padding = "same",
                kernel_initializer = "lecun_normal", use_bias = True
            )

    def call(self, x):
        shortcut = x if self.proj is None else self.proj(x)

        b1 = tf.nn.selu(self.b1(x))
        b3 = tf.nn.selu(self.b3(x))
        b5 = tf.nn.selu(self.b5(x))
        bp = tf.nn.selu(self.bp(self.pool(x)))

        y = (b1 + b3 + b5 + bp) * self.inv_sqrt4

        g = tf.nn.selu(self.gate(x))
        g = tf.exp(self.gate_scale * g)

        y = y * g

        return self.inv_sqrt2 * (shortcut + y)

#PIPELINE AND ALL PIPLINE PROBLEMS ARE IN THE GENERATOR AND PREPROCESSING
def parquet_generator(table):
    for idx, row in enumerate(table.to_pylist()):
        img = row["image"]
        label = row["label"]

        if isinstance(img, dict):
            if "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
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

        img = img / 255.0

        yield img, np.int32(label)

def preprocess(x, y):
    x = tf.image.resize(x, (128, 128))
    x = tf.image.per_image_standardization(x)
    return x, y

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataloc = os.path.join(base_dir, 'data')
    table = pq.read_table(dataloc)
    pathsave = os.path.join(base_dir, "weights", "MRISELUresin.keras")

    num_samples = table.num_rows

    dataset = tf.data.Dataset.from_generator(lambda: parquet_generator(table), output_signature = (tf.TensorSpec(shape = (None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int32),),)
    dataset = dataset.shuffle(num_samples, seed = 67, reshuffle_each_iteration = False)

    train_size = int(0.8 * num_samples)
    train = dataset.take(train_size)
    test  = dataset.skip(train_size)

    train = (train.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))
    test = (test.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))

    labels = table.column("label").to_numpy()
    num_classes = int(labels.max() + 1)

    #model arch
    def build_model(num_classes):
        inputs = Input(shape = (128, 128, 3))

        x = Conv2D(128, 3, activation = 'selu', padding = "same", kernel_initializer = "lecun_normal")(inputs)

        x = ResSELU(128)(x)
        x = ResSELU(128, stride=2)(x)
        x = SELUInception(128)(x)

        x = ResSELU(128)(x)
        x = ResSELU(128, stride=2)(x)
        x = SELUInception(128)(x)

        x = ResSELU(256)(x)
        x = ResSELU(256, stride=2)(x)
        x = SELUInception(256)(x)

        x = GlobalAveragePooling2D()(x)

        outputs = Dense(num_classes, kernel_initializer="lecun_normal")(x)

        return Model(inputs, outputs)

    labels = table.column("label").to_numpy()
    num_classes = len(np.unique(labels))
    model = build_model(num_classes)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    ES = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.0001,
        patience = 16,
        verbose = 1,
        mode = "auto",
        restore_best_weights = True,
    )
    MC = ModelCheckpoint(
        filepath = pathsave,
        monitor = "val_loss",
        save_best_only = True,
        verbose = 1,
        save_freq = "epoch",

    )

    steps_per_epoch = train_size // 32
    validation_steps = (num_samples - train_size) // 32

    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-3, momentum = 0.9, nesterov = True)

    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    model.fit(train, epochs = 256, validation_data = test, steps_per_epoch=steps_per_epoch, callbacks = [ES, MC],)

if __name__ == "__main__":
    main()