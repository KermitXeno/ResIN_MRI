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
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyarrow.parquet as pq
from GNNpkg.SELUselfnorm import SELUInception, SELUResidual, lecun_lr

def parquet_generator(table):
    for row in table.to_pylist():

        img = row["image"]
        label = row["label"]

        if isinstance(img, dict):
            if "bytes" in img:
                img = Image.open(io.BytesIO(img["bytes"])).convert("RGB")

            elif "data" in img and "shape" in img:
                arr = np.array(img["data"], dtype=np.uint8).reshape(img["shape"])
                img = Image.fromarray(arr)

            else:
                raise ValueError(f"Unknown image dict format: {list(img.keys())}")

        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img)).convert("RGB")

        else:
            img = Image.fromarray(np.asarray(img))

        img = np.array(img, dtype=np.float32)

        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)

        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = img / 255.0

        yield img, np.int32(label)

def build_model(num_classes):
    inputs = Input(shape=(128, 128, 3))

    x = Conv2D(64, 3, padding = "same", kernel_initializer = "lecun_normal", use_bias = False)(inputs)

    x = SELUResidual(64)(x)
    x = SELUResidual(64, stride=2)(x)
    x = SELUInception(64)(x)

    x = SELUResidual(128)(x)
    x = SELUResidual(128, stride=2)(x)
    x = SELUInception(64)(x)

    x = GlobalAveragePooling2D()(x)

    outputs = Dense(num_classes, kernel_initializer = "lecun_normal")(x)

    return Model(inputs, outputs)

def trainResinSELU():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataloc = os.path.join(base_dir, "data")
    pathsave = os.path.join(base_dir, "weights", "MRISELUresin.keras")
    os.makedirs(os.path.join(base_dir, "weights"), exist_ok=True)

    table = pq.read_table(dataloc)

    num_samples = table.num_rows
    labels = table.column("label").to_numpy()
    num_classes = int(labels.max() + 1)
    BATCH = 32
    train_size = int(0.8 * num_samples)

    dataset = tf.data.Dataset.from_generator(
        lambda: parquet_generator(table),
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    dataset = dataset.shuffle(num_samples, seed=67, reshuffle_each_iteration=False)
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size)
    steps_per_epoch = train_size // BATCH
    val_steps = (num_samples - train_size) // BATCH

    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(2048, reshuffle_each_iteration=True)
                .batch(BATCH)
                .repeat()
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH)
              .prefetch(tf.data.AUTOTUNE))

    model = build_model(num_classes)
    eta_sgd = min(1e-3, lecun_lr(D=2, L_ell=1.0) / 10.0)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=eta_sgd, momentum=0.9, nesterov=True, clipnorm=1.0)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=16,
                      verbose=1, restore_best_weights=True),

        ModelCheckpoint(filepath=pathsave, monitor="val_loss",
                        save_best_only=True, verbose=1, save_freq="epoch"),
    ]

    model.fit(train_ds, epochs=256, validation_data=val_ds,
              steps_per_epoch=steps_per_epoch, validation_steps=val_steps,
              callbacks=callbacks)

if __name__ == "__main__":
    trainResinSELU()