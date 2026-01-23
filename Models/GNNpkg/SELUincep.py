# SELU Inception Module for TensorFlow/Keras : this module is retired and needs to be changed somehow
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D

class SELUInception(tf.keras.layers.Layer):
    def __init__(self, channels, scale = 0.1):
        super().__init__()
        self.channels = channels
        self.gate_scale = scale
        self.inv_sqrt2 = tf.constant(1.0 / tf.sqrt(2.0), tf.float32)
        self.inv_sqrt4 = tf.constant(1.0 / tf.sqrt(4.0), tf.float32)

        self.b1 = Conv2D(channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)
        self.b3 = Conv2D(channels, 3, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)
        self.b5 = Conv2D(channels, 5, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)

        self.pool = MaxPooling2D(3, strides = 1, padding = "same")
        self.bp = Conv2D(channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias=True)

        self.gate = Conv2D(channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)

        self.proj = None

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.proj = Conv2D(self.channels, 1, padding = "same", kernel_initializer = "lecun_normal", use_bias = True)

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
