import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AlphaDropout, MaxPooling2D, BatchNormalization, ReLU


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
