
import os
import sys
import tensorflow as tf
from tensorflow import keras
#from Models.GNNpkg.RELUres import ResRELU
#from Models.GNNpkg.RELUincep import RELUInception
from Models.Train_MRI_ResInRELU import build_model
from Models.Train_MRI_TransferRes import trainResnet50v2

#CALL THIS AT START OF APP.PY
def initialize():
    UTILSp = os.path.dirname(os.path.abspath(__file__))
    MRIp   = os.path.dirname(UTILSp)

    MODELp = os.path.join(MRIp, "Models")

    if not os.path.exists(os.path.join(MODELp, "weights")):
        os.makedirs(os.path.join(MODELp, "weights"))

    MODELp = os.path.join(MODELp, "weights")

    tpath = os.path.join(MODELp, "MRIresnet50v2.weights.h5")
    rpath = os.path.join(MODELp, "MRIRELUresin.weights.h5")

    return tpath, rpath

def initRELUtrans(tpath):
    model = trainResnet50v2(4)
    model.load_weights(tpath)

    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5e-4, momentum=0.9)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


#selu is not currently supported in tf.keras.models.load_model
def initSELURES():
    pass

def initRELURES(rpath):
    model = build_model(4)
    model.load_weights(rpath)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.05, momentum=0.9, nesterov=True
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    initialize()