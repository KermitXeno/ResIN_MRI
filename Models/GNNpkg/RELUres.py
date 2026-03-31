#RESNET style residual block with ReLU activations

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense

class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, survival_prob, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_prob

    def call(self, x, training=None):
        if training is False or self.survival_prob == 1.0:
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
        if training is False or self.keep_prob == 1.0:
            return x

        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]

        def no_drop():
            return x

        def apply_dropblock():
            valid_h = tf.maximum(h - self.block_size + 1, 1)
            valid_w = tf.maximum(w - self.block_size + 1, 1)

            gamma = (
                (1.0 - self.keep_prob) * tf.cast(h * w, tf.float32)
                / 
                tf.cast(self.block_size ** 2 * valid_h * valid_w, tf.float32)
            )

            mask = tf.cast(tf.random.uniform([tf.shape(x)[0], h, w, c]) < gamma,tf.float32,)

            mask = -tf.nn.max_pool2d(-mask, ksize = self.block_size, strides = 1, padding = "VALID",)

            pad_h = h - tf.shape(mask)[1]
            pad_w = w - tf.shape(mask)[2]
            mask = tf.pad(mask, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

            keep = 1.0 - mask

            keep_mean = tf.reduce_mean(keep, axis=[1, 2, 3], keepdims=True)
            keep = keep / tf.maximum(keep_mean, 1e-6)

            return x * keep

        return tf.cond(tf.logical_or(h < self.block_size, w < self.block_size), no_drop, apply_dropblock,)

class SqueezeExcitation(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction

        self.gap = GlobalAveragePooling2D(keepdims=True)

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        bottleneck = max(1, in_channels // self.reduction)

        self.fc1 = Dense(bottleneck, activation="relu", use_bias=True)
        self.fc2 = Dense(in_channels, activation="sigmoid", use_bias=True)

        self.fc1.build((None, 1, 1, in_channels))
        self.fc2.build((None, 1, 1, bottleneck))

        super().build(input_shape)

    def call(self, x):
        scale = self.gap(x)
        scale = self.fc1(scale)
        scale = self.fc2(scale)
        return x * scale


class ResRELU(tf.keras.layers.Layer):
    
    def __init__(self, channels, stride = 1, sereduction = 16, probs1 = 1.0, probs2 = 0.9, keeps1 = 1.0, keeps2 = 0.9, blocksize = 7, **kwargs,):
        super().__init__(**kwargs)
        self.out_channels = channels
        self.stride = stride
        mid_channels = channels // 4

        self.conv1 = Conv2D(mid_channels, 1, strides=stride, padding='same', use_bias=False, activation="relu", kernel_initializer='he_normal')
        self.se1 = SqueezeExcitation(mid_channels, reduction = max(1, sereduction // 4))
        self.db1 = DropBlock2D(blocksize, keeps1)
        self.sd1 = StochasticDepth(probs1)

        self.conv2 = Conv2D(mid_channels, 3, strides=1, padding='same', use_bias=False, activation="relu", kernel_initializer='he_normal')
        self.se2 = SqueezeExcitation(mid_channels, reduction = max(1, sereduction // 4))
        self.db2 = DropBlock2D(blocksize, keeps2)
        self.sd2 = StochasticDepth(probs2)

        self.conv3 = Conv2D(channels, 1, strides=1, padding='same', use_bias=False, activation="relu", kernel_initializer='he_normal')
        self.se3 = SqueezeExcitation(channels, reduction = sereduction)

        self.shortcut_conv = None

    def build(self, input_shape):
        if input_shape is None or input_shape[-1] is None:
            return

        if self.stride != 1 or input_shape[-1] != self.out_channels:
            self.shortcut_conv = Conv2D(self.out_channels, 1, strides = self.stride, padding = 'same', use_bias = False, kernel_initializer = 'he_normal')
            self.shortcut_bn = BatchNormalization()
        super().build(input_shape)

    def call(self, x, training = None):
        if x is None:
            raise ValueError("ResRELU received None as input")

        kw = dict(training=training)

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(x), **kw)
        else:
            shortcut = x

        y = self.conv1(x, **kw)
        y = self.se1(y)
        y = self.db1(y, **kw)
        y = self.sd1(y, **kw)

        y = self.conv2(y, **kw)
        y = self.se2(y)
        y = self.db2(y, **kw)
        y = self.sd2(y, **kw)

        y = self.conv3(y, **kw)
        y = self.se3(y)

        return shortcut + y