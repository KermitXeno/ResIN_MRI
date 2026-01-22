# SELU Inception Module for TensorFlow/Keras : this module is retired and needs to be changed somehow

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AlphaDropout


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
