# -*- coding: utf-8 -*-
"""
@author: elamr

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
from tensorflow.keras import regularizers
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import os
import io

base_dir = os.path.dirname(os.path.abspath(__file__))

class BottleneckSELU(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride = 1):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

        self.act1 = Activation("selu")
        self.conv1 = Conv2D(out_channels, 1, kernel_initializer = "lecun_normal")

        self.act2 = Activation("selu")
        self.conv2 = Conv2D(out_channels, 3, strides = stride,
                            padding = "same", kernel_initializer = "lecun_normal")

        self.act3 = Activation("selu")
        self.conv3 = Conv2D(out_channels, 1, kernel_initializer = "lecun_normal")

        self.shortcut = None

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.out_channels:
            self.shortcut = Conv2D(
                self.out_channels, 1, strides = self.stride,
                padding = "same", kernel_initializer = "lecun_normal"
            )

    def call(self, x):
        y = self.act1(x)
        y = self.conv1(y)
        y = self.act2(y)
        y = self.conv2(y)
        y = self.act3(y)
        y = self.conv3(y)

        shortcut = x if self.shortcut is None else self.shortcut(x)
        return shortcut + 0.1 * y

class SELUInception(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"

        branch_channels = out_channels // 4

        self.b1 = Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same")

        self.b2 = Sequential([
            Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same"),
            DepthwiseConv2D(3, padding = "same", depthwise_initializer = "lecun_normal"),
            Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same")
        ])

        self.b3 = Sequential([
            Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same"),
            DepthwiseConv2D(5, padding="same", depthwise_initializer = "lecun_normal"),
            Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same")
        ])

        self.b4 = Sequential([
            AveragePooling2D(pool_size = 3, strides = 1, padding = "same"),
            Conv2D(branch_channels, 1, activation = "selu", kernel_initializer = "lecun_normal", padding = "same")
        ])

        self.residual = None

    def build(self, input_shape):
        in_channels = input_shape[-1]
        if in_channels != self.out_channels:
            self.residual = Conv2D(self.out_channels, 1, padding = "same", kernel_initializer = "lecun_normal")

    def call(self, x):

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)

        out = tf.concat([y1, y2, y3, y4], axis = -1)

        if self.residual is not None:
            x = self.residual(x)

        return out + 0.1 * out 

def main():

    dataloc = os.path.join(base_dir, 'data')
    table = pq.read_table(dataloc)
    pathsave = os.path.join(base_dir, 'weights', 'AMRI.keras')

    num_samples = table.num_rows

    def parquet_generator():
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

            img = (img - 127.5) / 127.5

            yield img, np.int32(label)

    dataset = tf.data.Dataset.from_generator(
    parquet_generator,
    output_signature = (tf.TensorSpec(shape = (None, None, 3), dtype = tf.float32),tf.TensorSpec(shape = (), dtype=tf.int32),),)

    dataset = dataset.shuffle(num_samples, seed=67, reshuffle_each_iteration = False)

    train_size = int(0.8 * num_samples)
    train = dataset.take(train_size)
    test  = dataset.skip(train_size)

    def preprocess(x, y):
        x = tf.image.resize(x, (128, 128))
        return x, y

    train = (train.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).repeat().prefetch(tf.data.AUTOTUNE))
    test = (test.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))

    #model arch
    def build_model(num_classes):
        inputs = Input(shape = (128, 128, 3))

        x = Conv2D(64, 3, activation = 'selu', padding = "same", kernel_initializer = "lecun_normal")(inputs)

        x = BottleneckSELU(64)(x)
        x = BottleneckSELU(64, stride=2)(x)
        x = SELUInception(64)(x)

        x = BottleneckSELU(128)(x)
        x = BottleneckSELU(128, stride=2)(x)
        x = SELUInception(128)(x)

        x = BottleneckSELU(256)(x)
        x = BottleneckSELU(256, stride=2)(x)
        x = SELUInception(256)(x)

        x = AlphaDropout(0.15)(x)

        x = GlobalAveragePooling2D()(x)

        outputs = Dense(num_classes, activation="softmax", kernel_initializer="lecun_normal")(x)

        return Model(inputs, outputs)

    labels = table.column("label").to_numpy()
    num_classes = int(labels.max() + 1)
    model = build_model(num_classes)

    ES = EarlyStopping(
        monitor = "val_loss",
        min_delta = 0.01,
        patience = 16,
        verbose = 1,
        mode = "auto",
        restore_best_weights = True,
        start_from_epoch = 0,
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
    optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    model.fit(train, epochs = 128, steps_per_epoch = steps_per_epoch, validation_data = test, validation_steps = validation_steps, callbacks = [ES, MC],)

    model.save(pathsave)
 
if __name__ == "__main__":
    main()