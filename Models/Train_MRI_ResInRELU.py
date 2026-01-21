# -*- coding: utf-8 -*-
"""
@author: elamr

This is a custom implementation of a Residual Network with RELU and Inception blocks for Alzheimer's MRI
classification that uses modern regularization techniques like Stochastic Depth and DropBlock.

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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import os
import io

#ALL RES BLOCK COMPONENTS
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, survival_prob, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, x, training=None):
        if not training or self.survival_prob == 1.0:
            return x
        batch_size = tf.shape(x)[0]
        random_tensor = self.survival_prob + tf.random.uniform([batch_size, 1, 1, 1])
        binary_tensor = tf.floor(random_tensor)
        return (x / self.survival_prob) * binary_tensor

class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, block_size, keep_prob, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob

    def call(self, x, training=None):
        if not training or self.keep_prob == 1.0:
            return x

    # Spatial dimensions
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]

    # If feature map is smaller than block_size, skip DropBlock
        def no_drop():
            return x

        def apply_dropblock():
            # Compute valid region sizes
            valid_h = tf.maximum(h - self.block_size + 1, 1)
            valid_w = tf.maximum(w - self.block_size + 1, 1)

            gamma = (
                (1.0 - self.keep_prob) * tf.cast(h * w, tf.float32)
                / 
                tf.cast(self.block_size ** 2 * valid_h * valid_w, tf.float32)
            )

            # Sample mask
            mask = tf.cast(tf.random.uniform([tf.shape(x)[0], h, w, c]) < gamma,tf.float32,)

            # Create blocks
            mask = -tf.nn.max_pool2d(-mask, ksize = self.block_size, strides = 1, padding = "VALID",)

            pad_h = h - tf.shape(mask)[1]
            pad_w = w - tf.shape(mask)[2]
            mask = tf.pad(mask, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

            keep = 1.0 - mask

            # Normalize activations
            keep_mean = tf.reduce_mean(keep, axis=[1, 2, 3], keepdims=True)
            keep = keep / tf.maximum(keep_mean, 1e-6)

            return x * keep

        return tf.cond(tf.logical_or(h < self.block_size, w < self.block_size), no_drop, apply_dropblock,)

class ResRELU(tf.keras.layers.Layer):
    def __init__(self, channels, stride = 1, reduction = 16, survival_prob = 1.0, dropblock_keep_prob = 1.0, block_size = 7, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = channels
        self.stride = stride
        self.mid_channels = channels // 4

        # Pre-activation
        self.bn1 = BatchNormalization()
        self.relu1 = ReLU()

        # Bottleneck convs
        self.conv1 = Conv2D(self.mid_channels, 1, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')
        self.bn2 = BatchNormalization()
        self.relu2 = ReLU()

        self.conv2 = Conv2D(self.mid_channels, 3, strides = stride, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')
        self.bn3 = BatchNormalization()
        self.relu3 = ReLU()

        self.conv3 = Conv2D(self.out_channels, 1, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')

        # Squeeze-and-Excitation
        self.global_pool = GlobalAveragePooling2D()
        se_width = max(1, self.mid_channels // reduction)
        self.se_reduce = Dense(se_width, activation='relu')
        self.se_expand = Dense(self.out_channels, activation = 'sigmoid')

        # Regularizers
        self.dropblock = DropBlock2D(block_size, dropblock_keep_prob)
        self.stoch_depth = StochasticDepth(survival_prob)

        # Shortcut
        self.shortcut_conv = None

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.out_channels:
            self.shortcut_conv = Conv2D(self.out_channels, 1, strides = self.stride, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')
            self.shortcut_bn = BatchNormalization()
        super().build(input_shape)

    def call(self, x, training = None):
        # Pre-activation
        y = self.bn1(x, training = training)
        y = self.relu1(y)

        # Shortcut path
        if self.shortcut_conv:
                shortcut = self.shortcut_bn(self.shortcut_conv(x), training=training)
        else:
            shortcut = x

        # Bottleneck conv path
        y = self.conv1(y)
        y = self.bn2(y, training = training)
        y = self.relu2(y)

        y = self.conv2(y)
        y = self.bn3(y, training = training)
        y = self.relu3(y)

        y = self.conv3(y)

        # Squeeze-and-Excitation here when i find a good implementation
       
        y = Multiply()([y, se])

        # Stochastic Depth
        y = self.stoch_depth(y, training = training)

        # DropBlock
        y = self.dropblock(y, training = training)

        # Final merge
        return shortcut + y

#INCPETION BLOCK
class RELUInception(tf.keras.layers.Layer):
    def __init__(self, channels, gate_scale = 0.1):
        super().__init__()
        self.channels = channels
        self.gate_scale = gate_scale

        self.inv_sqrt2 = tf.constant(1.0 / tf.sqrt(2.0), tf.float32)
        self.inv_sqrt4 = tf.constant(1.0 / tf.sqrt(4.0), tf.float32)

        # Inception branches
        self.b1 = Conv2D(channels, 1, padding = "same", kernel_initializer = "he_normal", use_bias = False)
        self.b3 = Conv2D(channels, 3, padding = "same", kernel_initializer = "he_normal", use_bias = False)
        self.b5 = Conv2D(channels, 5, padding = "same", kernel_initializer = "he_normal", use_bias = False)

        self.pool = MaxPooling2D(3, strides = 1, padding = "same")
        self.bp = Conv2D(channels, 1, padding = "same",)

        # BatchNorm + ReLU shared pattern
        self.bn1 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn5 = BatchNormalization()
        self.bnp = BatchNormalization()

        # Gating branch
        self.gate = Conv2D(channels, 1, padding = "same", kernel_initializer = "he_normal", use_bias = False)
        self.gate_bn = BatchNormalization()

        self.relu = ReLU()
        self.proj = None

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.proj = Conv2D(self.channels, 1, padding = "same", kernel_initializer = "he_normal", use_bias = False)
        super().build(input_shape)

    def call(self, x, training=None):
        shortcut = x if self.proj is None else self.proj(x)

        # Inception branches
        b1 = self.relu(self.bn1(self.b1(x), training = training))
        b3 = self.relu(self.bn3(self.b3(x), training = training))
        b5 = self.relu(self.bn5(self.b5(x), training = training))
        bp = self.relu(self.bnp(self.bp(self.pool(x)), training = training))

        y = (b1 + b3 + b5 + bp) * self.inv_sqrt4

        # Gating
        g = self.gate_bn(self.gate(x), training=training)
        g = tf.sigmoid(self.gate_scale * g)

        y = y * g

        # Residual merge with variance control
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

        yield img, np.int32(label)

def preprocess(x, y):
    x = tf.image.resize(x, (128, 128))
    x = tf.image.per_image_standardization(x)
    return x, y

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataloc = os.path.join(base_dir, 'data')
    table = pq.read_table(dataloc)
    pathsave = os.path.join(base_dir, "weights", "MRIRELUresin.keras")

    num_samples = table.num_rows

    dataset = tf.data.Dataset.from_generator(lambda: parquet_generator(table), output_signature = (tf.TensorSpec(shape = (None, None, 3), dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int32),),)
    dataset = dataset.shuffle(num_samples, seed = 67, reshuffle_each_iteration = True)

    train_size = int(0.8 * num_samples)
    train = dataset.take(train_size)
    test  = dataset.skip(train_size)

    train = (train.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).repeat().prefetch(tf.data.AUTOTUNE))
    test = (test.map(preprocess, num_parallel_calls = tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE))

    labels = table.column("label").to_numpy()

    #model arch
    def build_model(num_classes):
        inputs = Input(shape = (128, 128, 3))

        x = Conv2D(64, 3, padding = "same", activation = 'relu', kernel_initializer = "he_normal", use_bias = False)(inputs)
        x = BatchNormalization()(x)


        x = ResRELU(64)(x)
        x = ResRELU(64, stride = 2)(x)
        x = RELUInception(64)(x)

        x = ResRELU(128)(x)
        x = ResRELU(128, stride = 2)(x)
        x = RELUInception(128)(x)

        x = ResRELU(256)(x)
        x = ResRELU(256, stride = 2)(x)
        x = RELUInception(256)(x)

        x = GlobalAveragePooling2D()(x)

        outputs = Dense(num_classes, kernel_initializer = "he_normal")(x)

        return Model(inputs, outputs)

    labels = table.column("label").to_numpy()
    num_classes = int(labels.max()) + 1
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
    val_steps = (num_samples - train_size) // 32

    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05, momentum = 0.9, nesterov = True)

    model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
    model.fit(train, epochs = 256, steps_per_epoch = steps_per_epoch, validation_data = test, validation_steps = val_steps, callbacks = [ES, MC],)

if __name__ == "__main__":
    main()