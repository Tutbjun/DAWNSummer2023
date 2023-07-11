#train tensorflow model to aproximate fsps model

import numpy as np
import os
from fspsCore import blackbox
import time
import multiprocessing as mp


settingSet = {
    'Z': [-5,-1],
    'IMFtemp': [8, 60],
    'zred': [0,20],
    'const': [0,0.5],
    'tau': [0.5,2],
    'sf_start': [0, 7],
    'sf_trunc': [9, 12],
    'fburst': [0,0.5],
    'tburst': [0,7],
    'mwr': [1,5],
    'uvb': [0.2,3],
    'tage': [0,13]
}

#start wirht the first few parameters and keep the others constant average
#create a random sampled set of these parameters (this is in set)
#run the fsps model for each set and create spectre
#run filterset on spectre (then this is out set)
#train tensorflow to get from in set to out set

#improve model by adding more parameters one by one

#asume the ussage of a sample method where the samplecount is the same as the axis fidelity
sampleCnt = 20000

initialVarCount = 3
for varCount in list(range(initialVarCount, len(settingSet))):
    if varCount > initialVarCount: raise NotImplementedError("not implemented yet")
    print("gening " + str(varCount) + "dimensions of variables")
    #create random sample of parameters
    settingSamples = []
    for k in list(settingSet.keys())[:varCount]:
        settingSamples.append(np.linspace(settingSet[k][0], settingSet[k][1], sampleCnt))

    Xs = [np.zeros(len(settingSet),dtype=np.float64) for i in range(sampleCnt)]
    sample2Pick = np.arange(sampleCnt)
    
    for i in range(sampleCnt):
        for l,k in enumerate(list(settingSet.keys())[:varCount]):
            j = np.random.choice(sample2Pick)
            Xs[i][l] = settingSamples[l][j]
            settingSamples[l] = np.delete(settingSamples[l], j)
        sample2Pick = sample2Pick[:-1]
        for l,k in enumerate(list(settingSet.keys())[varCount:]):
            Xs[i][l+varCount] = settingSet[k][0] + (settingSet[k][1] - settingSet[k][0])/2

    #run fsps model for each sample
    #run filterset on spectre
    #and for every 10 samples save the data
    np.save("Xs.npy", Xs)
    Ys = [[] for i in range(len(Xs))]
    #!multiprocessing
    def mp_worker(inputs):
        #print(mp.current_process().name)
        threadID = int(mp.current_process().name.split("-")[-1])-1
        X, i = inputs#unpack
        print("working on sample ", sampleCnt-i, "...")
        #print("i = ", i, "X = ", X)
        Ys[i] = blackbox(X, threadID)
        #print("i = ", i, "threadID = ", threadID)

        
    pool = mp.Pool(processes=8)
    X4pool = zip(Xs, range(len(Xs)))
    
    pool.map(mp_worker, X4pool)
    np.save(file="Xs_stage{varCount}_{time.time()}.npy", arr=Xs)
    np.save(file="Ys_stage{varCount}_{time.time()}.npy", arr=Ys)
    os.remove("Xs.npy")
    os.remove("Ys.npy")
