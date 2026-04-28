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
C_MAX_SQ = (_LAMBDA_SELU ** 2) * (1.0 + _ALPHA_SELU ** 2) / 2.0


def _branch_grad_correction(x: tf.Tensor, gamma: float = GAMMA) -> tf.Tensor:
    @tf.custom_gradient
    def _op(t):
        def _grad(dy):
            return dy * gamma
        return tf.identity(t), _grad
    return _op(x)


class OrthogonalLeCunInitializer(initializers.Initializer):
    def __init__(self, seed: int = None):
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        n_out = int(shape[-1])
        n_in = int(np.prod(shape[:-1]))

        rng = np.random.default_rng(self.seed)
        flat = rng.standard_normal((n_out, n_in)).astype(np.float32)

        u, _, vt = np.linalg.svd(flat, full_matrices=False)
        q = vt
        q = q / math.sqrt(n_in)

        return tf.cast(tf.reshape(q, shape), dtype)

    def get_config(self):
        return {"seed": self.seed}


class ZCAWhiten(Layer):
    def __init__(self, momentum: float = 0.99, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.momentum = momentum
        self.epsilon = epsilon

    def build(self, input_shape):
        c = int(input_shape[-1])
        self._c = c

        self.running_mean = self.add_weight(
            name="running_mean", shape=(c,),
            initializer="zeros", trainable=False
        )
        self.running_cov = self.add_weight(
            name="running_cov", shape=(c, c),
            initializer=tf.keras.initializers.Identity(),
            trainable=False
        )
        super().build(input_shape)

    def call(self, x, training=None):
        input_shape = tf.shape(x)
        c = self._c

        flat = tf.cast(tf.reshape(x, [-1, c]), tf.float32)
        n = tf.cast(tf.shape(flat)[0], tf.float32)

        if training:
            mu = tf.reduce_mean(flat, axis=0)
            centered = flat - mu
            cov = tf.matmul(centered, centered, transpose_a=True) / (n - 1.0 + self.epsilon)

            self.running_mean.assign(
                self.momentum * self.running_mean + (1.0 - self.momentum) * mu
            )
            self.running_cov.assign(
                self.momentum * self.running_cov + (1.0 - self.momentum) * cov
            )

        centered = flat - self.running_mean

        eigvals, eigvecs = tf.linalg.eigh(self.running_cov)
        eigvals = tf.maximum(eigvals, self.epsilon)
        inv_sqrt = 1.0 / tf.sqrt(eigvals)

        W_zca = eigvecs * inv_sqrt[tf.newaxis, :]
        W_zca = tf.matmul(W_zca, eigvecs, transpose_b=True)
        W_zca = tf.stop_gradient(W_zca)

        whitened = tf.matmul(centered, W_zca, transpose_b=True)
        return tf.reshape(whitened, input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"momentum": self.momentum, "epsilon": self.epsilon})
        return cfg


class SELUResidual(Layer):
    def __init__(self, out_channels: int, stride: int = 1, dropout_rate: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.stride = stride
        self.dropout_rate = dropout_rate

        self.conv1 = Conv2D(
            out_channels, 3, padding="same",
            kernel_initializer="lecun_normal",
            use_bias=False
        )
        self.conv2 = Conv2D(
            out_channels, 3, strides=stride, padding="same",
            kernel_initializer="lecun_normal",
            use_bias=False
        )

        self.dropout = AlphaDropout(dropout_rate)
        self._shortcut_proj = None

    def build(self, input_shape):
        in_c = int(input_shape[-1])
        if self.stride != 1 or in_c != self.out_channels:
            self._shortcut_proj = Conv2D(
                self.out_channels, 1, strides=self.stride, padding="same",
                kernel_initializer="lecun_normal",
                use_bias=False
            )
        super().build(input_shape)

    def call(self, x, training=None):
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
        cfg.update({
            "out_channels": self.out_channels,
            "stride": self.stride,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


class SELUInception(Layer):
    def __init__(self, channels: int, zca_momentum: float = 0.99, dropout_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.zca_momentum = zca_momentum
        self.dropout_rate = dropout_rate

        _init = OrthogonalLeCunInitializer()

        self.b1x1 = Conv2D(channels, 1, padding="same", kernel_initializer=_init, use_bias=False)
        self.b3x3 = Conv2D(channels, 3, padding="same", kernel_initializer=_init, use_bias=False)
        self.b5x5 = Conv2D(channels, 5, padding="same", kernel_initializer=_init, use_bias=False)
        self.bpool = Conv2D(channels, 1, padding="same", kernel_initializer=_init, use_bias=False)

        self.zca = ZCAWhiten(momentum=zca_momentum)
        self.dropout = AlphaDropout(dropout_rate) if dropout_rate > 0 else None
        self._pool = None
        self._skip = None

    def build(self, input_shape):
        in_c = int(input_shape[-1])
        out_c = 4 * self.channels

        self._pool = MaxPooling2D(3, strides=1, padding="same")

        if in_c != out_c:
            self._skip = Conv2D(
                out_c, 1, padding="same",
                kernel_initializer="lecun_normal",
                use_bias=False
            )
        super().build(input_shape)

    def call(self, x, training=None):
        b1 = tf.nn.selu(self.b1x1(x))
        b3 = tf.nn.selu(self.b3x3(x))
        b5 = tf.nn.selu(self.b5x5(x))
        bp = tf.nn.selu(self.bpool(self._pool(x)))

        y = tf.concat([b1, b3, b5, bp], axis=-1)

        if self.dropout is not None:
            y = self.dropout(y, training=training)

        y = self.zca(y, training=training)
        y = _branch_grad_correction(y, gamma=GAMMA)

        s = x if self._skip is None else self._skip(x)

        return BETA * y + BETA * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "channels": self.channels,
            "zca_momentum": self.zca_momentum,
            "dropout_rate": self.dropout_rate,
        })
        return cfg


def block_contraction_rate(D: int) -> float:
    return (1.0 + C_V ** D) / 2.0


def lecun_lr(D: int, L_ell: float = 1.0) -> float:
    kappa = block_contraction_rate(D)
    eta_s = 1.0 / (2.0 * L_ell * C_MAX_SQ ** (D / 2.0))
    eta_kappa = 3.0 * (1.0 - kappa ** 2) ** 2 / 8.0
    eta_q = 2.0 / Q
    return min(eta_s, eta_kappa, eta_q)