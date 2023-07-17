#take the X and Y arrays
#and segment them into a 1:9 pile
#call one train and the other test

testFrac = 0.1

import os
import numpy as np

dir = os.listdir()
dir = [d for d in dir if "Xs_stage" in d or "Ys_stage" in d]
times = [d.split("_")[-1].split(".npy")[0] for d in dir]
ind = [i for i,d in enumerate(dir) if "Xs_" in d]
Xs = list(np.array(dir)[np.array(ind)])
Xs_times = list(np.array(times)[np.array(ind)].astype(float))
ind = [i for i,d in enumerate(dir) if "Ys_" in d]
Ys = list(np.array(dir)[np.array(ind)])
Ys_times = list(np.array(times)[np.array(ind)].astype(float))

latestX = max(Xs_times)
latestY = max(Ys_times)
if latestX != latestY: raise Exception
ind = np.where(Xs_times==latestX)[0][0]
del Xs_times, Ys_times, times, dir, latestX, latestY
Xs = np.load(Xs[ind])
Ys = np.load(Ys[ind])
del ind

np.random.shuffle(sampling := np.arange(len(Xs)))
testInds = sampling[:int(len(Xs)*testFrac)]
trainInds = sampling[int(len(Xs)*testFrac):]

train_Xs = Xs[trainInds]
train_Ys = Ys[trainInds]
test_Xs = Xs[testInds]
test_Ys = Ys[testInds]

np.save("train_Xs.npy", train_Xs)
np.save("train_Ys.npy", train_Ys)
np.save("test_Xs.npy", test_Xs)
np.save("test_Ys.npy", test_Ys)