import tensorflow as tf
import tensorboard
import numpy as np
import os

#load data
train_Xs = np.load("train_Xs.npy")
train_Ys = np.load("train_Ys.npy")
test_Xs = np.load("test_Xs.npy")
test_Ys = np.load("test_Ys.npy")

#load model
dir = [d for d in os.listdir() if ".h5" in d]
dir = [d for d in dir if "model_" in d]
times = [d.split("_")[-1].split(".h5")[0] for d in dir]
ind = np.where(np.array(times)==max(times))[0][0]
dir = dir[ind]
model = tf.keras.models.load_model(dir)

#train model
model.fit(
    train_Xs, 
    train_Ys, 
    epochs=10, 
    validation_data=(test_Xs, test_Ys)
)
""",
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs")],
    verbose=1"""

#save model
model.save("model.h5")