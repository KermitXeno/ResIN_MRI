import tensorflow as tf

class GradCAM:
    def __init__(self, model, targetlayer):

        self.model = model
        self.targetlayer = targetlayer

        self.gradmodel = tf.keras.models.Model(
            inputs = model.inputs,
            outputs = [model.get_layer(targetlayer).output, model.output])

    def __call__(self, imagetensor, classidx = None):

        with tf.GradientTape() as tape:
            convout, preds = self.gradmodel(imagetensor, training = False)

            if classidx is None:
                classidx = tf.argmax(preds[0])

            score = preds[:, classidx]

        grads = tape.gradient(score, convout)

        weights = tf.reduce_mean(grads, axis = (1, 2))
        cam = tf.reduce_sum(convout * weights[:, None, None, :], axis = -1)

        cam = tf.nn.relu(cam)
        cam /= tf.reduce_max(cam) + 1e-8

        return cam[0].numpy()