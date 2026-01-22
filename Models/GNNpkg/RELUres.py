import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply

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

        #Squeeze and exitation should be here with new implementation class

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

#TODO Squeeze and exitation implementation here when i find a good class