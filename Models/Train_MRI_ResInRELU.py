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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyarrow.parquet as pq
import numpy as np
from PIL import Image
import os
import io
from GNNpkg.RELUres import ResRELU
from GNNpkg.RELUincep import InceptionRes

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
        x = InceptionRes(128, scale = 0.2)(x)

        x = ResRELU(128)(x)
        x = ResRELU(128, stride = 2)(x)
        x = InceptionRes(256, scale = 0.3)(x)

        x = ResRELU(256)(x)
        x = ResRELU(256, stride = 2)(x)
        x = InceptionRes(256, scale = 0.4)(x)

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