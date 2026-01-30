
import os
import tensorflow as tf
from tensorflow import keras

def initialize():
    UTILSp = os.path.dirname(os.path.abspath(__file__))
    MRIp   = os.path.dirname(UTILSp)

    MODELp = os.path.join(MRIp, "Models")

    if not os.path.exists(os.path.join(MODELp, "weights")):
        os.makedirs(os.path.join(MODELp, "weights"))

    MODELp = os.path.join(MODELp, "weights")

    global tpath, rpath
    tpath = os.path.join(MODELp, "MRIresnet50v2.keras")
    rpath = os.path.join(MODELp, "MRIRELUresin.keras")

def initRELUtrans():
    model = tf.keras.models.load_model(tpath, compile=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1.5e-4, momentum = 0.9)
    model.compile(optimizer = optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    return model

#selu is not currently supported in tf.keras.models.load_model
def initSELURES():
    pass

def initRELURES():
    model = tf.keras.models.load_model(rpath, compile=False)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.05, momentum = 0.9, nesterov = True)
    model.compile(optimizer = optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
    return model

if __name__ == "__main__":
    initialize()