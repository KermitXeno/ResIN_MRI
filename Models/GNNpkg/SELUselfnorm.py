import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, MaxPooling2D, AlphaDropout
from tensorflow.keras import initializers

_ALPHA_SELU = 1.6732632423543772
_LAMBDA_SELU = 1.0507009873554805
Q = 1.072
BETA = 1.0 / math.sqrt(2.0)
GAMMA = 1.0 / math.sqrt(Q)
C_V = 0.783
# The _branch_grad_correction function is a custom gradient operation that scales the gradients by a factor of gamma during backpropagation. 
# This is used to correct for the variance changes introduced by the SELU activation and the residual connections, helping to maintain 
# self-normalizing properties in the network. By applying this correction, we can ensure that the gradients do not explode or vanish as they 
# propagate through the layers, which is crucial for training deep networks with SELU activations effectively.
def _branch_grad_correction(x, gamma=GAMMA):
    @tf.custom_gradient
    def _op(t):
        def _grad(dy):
            return dy * gamma
        return tf.identity(t), _grad
    return _op(x)

# This initializer generates orthogonal weight matrices with a scaling factor that matches the variance of the LeCun normal 
# initializer, which is optimal for SELU activations. For layers where the fan-in is greater than or equal to the fan-out, it 
# creates a square orthogonal matrix and then slices it to the desired shape. For layers where the fan-out is greater than the fan-in, 
# it falls back to a scaled normal distribution. This approach helps maintain self-normalizing properties in deep networks using 
# SELU activations.
class OrthogonalLeCunInitializer(initializers.Initializer):
    def __init__(self, seed=None):
        self.seed = seed
    def __call__(self, shape, dtype=tf.float32):
        if len(shape) < 2:
            raise ValueError(f"Shape must be at least 2D, got {shape}")
        fan_in = int(np.prod(shape[:-1]))
        fan_out = int(shape[-1])
        rng = np.random.default_rng(self.seed)
        if fan_in >= fan_out:
            a = rng.standard_normal((fan_in, fan_in)).astype(np.float32)
            Q, _ = np.linalg.qr(a)
            W_2d = Q[:fan_in, :fan_out]
        else:
            std = 1.0 / math.sqrt(fan_in)
            W_2d = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std
        return tf.cast(tf.reshape(W_2d, shape), dtype)
    def get_config(self):
        return {"seed": self.seed}

# This layer implements ZCA whitening with a running mean and covariance for use in self-normalizing networks. It uses a full 
# covariance matrix when the number of channels is small, and a diagonal approximation when the number of channels is large, 
# to save memory and computation. The whitening is applied per batch during training, and the running statistics are used during 
# inference. The layer also includes safeguards against non-finite values and extreme outliers.
class ZCAWhiten(Layer):
    def __init__(self, momentum=0.99, epsilon=1e-5, max_full_channels=128, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon
        self.max_full_channels = max_full_channels
    def build(self, input_shape):
        c = int(input_shape[-1])
        self._c = c
        self._full = (c <= self.max_full_channels)
        self.running_mean = self.add_weight(
            name="running_mean", shape=(c,), initializer="zeros", trainable=False)
        if self._full:
            self.running_cov = self.add_weight(
                name="running_cov", shape=(c, c),
                initializer=tf.keras.initializers.Identity(), trainable=False)
        else:
            self.running_var = self.add_weight(
                name="running_var", shape=(c,), initializer="ones", trainable=False)
        super().build(input_shape)
    def call(self, x, training=None):
        c = self._c
        orig_shape = tf.shape(x)
        flat = tf.cast(tf.reshape(x, [-1, c]), tf.float32)
        flat = tf.where(tf.math.is_finite(flat), flat, tf.zeros_like(flat))
        flat = tf.clip_by_value(flat, -50.0, 50.0)
        n = tf.cast(tf.shape(flat)[0], tf.float32)
        if self._full:
            return self._full_zca(flat, orig_shape, n, training, x.dtype)
        return self._diag_zca(flat, orig_shape, training, x.dtype)
    def _full_zca(self, flat, orig_shape, n, training, dtype):
        if training:
            mu = tf.reduce_mean(flat, axis=0)
            centered = flat - mu
            cov = (tf.matmul(centered, centered, transpose_a=True)
                   / (n - 1.0 + self.epsilon)
                   + self.epsilon * tf.eye(self._c))
            cov = tf.where(tf.math.is_finite(cov), cov, tf.eye(self._c))
            self.running_mean.assign(self.momentum * self.running_mean + (1.0 - self.momentum) * mu)
            self.running_cov.assign(self.momentum * self.running_cov + (1.0 - self.momentum) * cov)
        centered = flat - self.running_mean
        eigvals, eigvecs = tf.linalg.eigh(self.running_cov)
        eigvals = tf.maximum(eigvals, self.epsilon)
        inv_sqrt = 1.0 / tf.sqrt(eigvals)
        W_zca = tf.stop_gradient(
            tf.matmul(eigvecs * inv_sqrt[tf.newaxis, :], eigvecs, transpose_b=True))
        whitened = tf.matmul(centered, W_zca, transpose_b=True)
        whitened = tf.where(tf.math.is_finite(whitened), whitened, tf.zeros_like(whitened))
        return tf.reshape(tf.cast(whitened, dtype), orig_shape)
    def _diag_zca(self, flat, orig_shape, training, dtype):
        if training:
            mu = tf.reduce_mean(flat, axis=0)
            var = tf.math.reduce_variance(flat, axis=0)
            var = tf.where(tf.math.is_finite(var), var, tf.ones_like(var))
            self.running_mean.assign(self.momentum * self.running_mean + (1.0 - self.momentum) * mu)
            self.running_var.assign(self.momentum * self.running_var + (1.0 - self.momentum) * var)
        centered = flat - self.running_mean
        std = tf.sqrt(tf.maximum(self.running_var, self.epsilon))
        whitened = centered / std
        whitened = tf.where(tf.math.is_finite(whitened), whitened, tf.zeros_like(whitened))
        return tf.reshape(tf.cast(whitened, dtype), orig_shape)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"momentum": self.momentum, "epsilon": self.epsilon,
                    "max_full_channels": self.max_full_channels})
        return cfg

# This layer implements a residual block with SELU activations and Alpha Dropout. It includes two convolutional layers, 
# optional downsampling, and a shortcut connection. The gradients are corrected using the _branch_grad_correction function to 
# maintain self-normalizing properties. The layer also handles non-finite values in the input and applies scaling to the output 
# to ensure stable training with SELU activations.
class SELUResidual(Layer):
    def __init__(self, out_channels, stride=1, dropout_rate=0.05, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.conv1 = Conv2D(out_channels, 3, padding="same",
                            kernel_initializer="lecun_normal", use_bias=False)
        self.conv2 = Conv2D(out_channels, 3, strides=stride, padding="same",
                            kernel_initializer="lecun_normal", use_bias=False)
        self.dropout = AlphaDropout(dropout_rate)
        self._shortcut_proj = None
    def build(self, input_shape):
        in_c = int(input_shape[-1])
        if self.stride != 1 or in_c != self.out_channels:
            self._shortcut_proj = Conv2D(self.out_channels, 1, strides=self.stride,
                                         padding="same", kernel_initializer="lecun_normal",
                                         use_bias=False)
        super().build(input_shape)
    def call(self, x, training=None):
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        h = tf.nn.selu(x)
        h = self.conv1(h)
        h = tf.nn.selu(h)
        h = self.conv2(h)
        h = self.dropout(h, training=training)
        h = _branch_grad_correction(h, gamma=GAMMA)
        s = x if self._shortcut_proj is None else self._shortcut_proj(x)
        return BETA * h + BETA * s
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_channels": self.out_channels, "stride": self.stride,
                    "dropout_rate": self.dropout_rate})
        return cfg

# This layer implements an Inception-like block with SELU activations and Alpha Dropout. It consists of four parallel branches: 1x1, 3x3, 5x5 convolutions, and a max pooling followed by a 1x1 convolution. 
# The outputs of the branches are concatenated and passed through ZCA whitening and gradient correction to maintain self-normalizing properties. The layer also includes a residual connection from the input to the output, 
# with optional projection if the number of channels changes. Non-finite values in the input are handled gracefully to ensure stable training.
class SELUInception(Layer):
    def __init__(self, channels, zca_momentum=0.99, zca_max_full_ch=128,
                 dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.zca_momentum = zca_momentum
        self.zca_max_full_ch = zca_max_full_ch
        self.dropout_rate = dropout_rate
        _init = OrthogonalLeCunInitializer()
        self.b1x1 = Conv2D(channels, 1, padding="same", kernel_initializer=_init, use_bias=False)
        self.b3x3 = Conv2D(channels, 3, padding="same", kernel_initializer=_init, use_bias=False)
        self.b5x5 = Conv2D(channels, 5, padding="same", kernel_initializer=_init, use_bias=False)
        self.bpool = Conv2D(channels, 1, padding="same", kernel_initializer=_init, use_bias=False)
        self.pool = MaxPooling2D(3, strides=1, padding="same")
        self.zca = ZCAWhiten(momentum=zca_momentum, max_full_channels=zca_max_full_ch)
        self.dropout = AlphaDropout(dropout_rate) if dropout_rate > 0 else None
        self._skip_proj = None
    def build(self, input_shape):
        in_c = int(input_shape[-1])
        out_c = 4 * self.channels
        if in_c != out_c:
            self._skip_proj = Conv2D(out_c, 1, padding="same",
                                     kernel_initializer="lecun_normal", use_bias=False)
        super().build(input_shape)
    def call(self, x, training=None):
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
        b1 = tf.nn.selu(self.b1x1(x))
        b3 = tf.nn.selu(self.b3x3(x))
        b5 = tf.nn.selu(self.b5x5(x))
        bp = tf.nn.selu(self.bpool(self.pool(x)))
        y = tf.concat([b1, b3, b5, bp], axis=-1)
        if self.dropout is not None:
            y = self.dropout(y, training=training)
        y = self.zca(y, training=training)
        y = _branch_grad_correction(y, gamma=GAMMA)
        s = x if self._skip_proj is None else self._skip_proj(x)
        return BETA * y + BETA * s
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "zca_momentum": self.zca_momentum,
                    "zca_max_full_ch": self.zca_max_full_ch, "dropout_rate": self.dropout_rate})
        return cfg

# The block_contraction_rate function calculates the contraction rate of a block in a self-normalizing network based on the depth D. It uses the constant C_V, 
# which is derived from the properties of the SELU activation function, to determine how much the activations contract as they pass through the layers. 
# The formula (1.0 + C_V ** D) / 2.0 provides an estimate of this contraction rate, which is crucial for understanding how the network maintains self-normalization 
# and for setting appropriate learning rates during training.
def block_contraction_rate(D):
    return (1.0 + C_V ** D) / 2.0

# The lecun_lr function calculates the learning rate for training a self-normalizing network based on the depth D and the Lipschitz constant L_ell of the loss function. It considers three factors:
def lecun_lr(D, L_ell=1.0):
    C_MAX_SQ = (_LAMBDA_SELU ** 2) * (1.0 + _ALPHA_SELU ** 2) / 2.0
    kappa = block_contraction_rate(D)
    eta_s = 1.0 / (2.0 * L_ell * C_MAX_SQ ** (D / 2.0))
    eta_kappa = 3.0 * (1.0 - kappa ** 2) ** 2 / 8.0
    eta_q = 2.0 / Q
    return min(eta_s, eta_kappa, eta_q)