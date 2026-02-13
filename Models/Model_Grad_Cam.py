import tensorflow as tf


class GradCAM:
    def __init__(self, model, targetlayer):
        self.model = model
        self.targetlayer = targetlayer

        self.feature_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer(targetlayer).output,
        )

    def __call__(self, imagetensor):

        features = self.feature_model(imagetensor, training=False)[0]

        h, w, c = features.shape

        reshaped = tf.reshape(features, (-1, c)) 

        reshaped -= tf.reduce_mean(reshaped, axis=0, keepdims=True)

        cov = tf.matmul(reshaped, reshaped, transpose_a=True)  # (C, C)

        eigvals, eigvecs = tf.linalg.eigh(cov)

        principal = eigvecs[:, -1]

        cam = tf.matmul(reshaped, principal[:, None]) 

        cam = tf.reshape(cam, (h, w))

        cam = tf.nn.relu(cam)
        maxv = tf.reduce_max(cam)
        if maxv > 0:
            cam = cam / maxv

        return cam.numpy()
