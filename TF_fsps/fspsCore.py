import os
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
import shutil
#import time as t
#import astropy.units as u
#from dust_extinction.parameter_averages import F04
#shutil.rmtree(matplotlib.get_cachedir())
#import csv
import fsps as sps
#from copy import copy
#import timeit

def setIMF(imfFile):
    SPS_HOME = os.getenv('SPS_HOME')
    imfPath = os.path.join(SPS_HOME, "data") #fsps install directory /data
    shutil.copyfile("imf_8-60k/"+imfFile,imfPath+"/imf.dat")

filterFuncs = []

def setFilters():
    #filterset in folder JWST_filters
    filterlist = os.listdir("JWST_filters")
    for f in filterlist:
        raw = np.loadtxt("JWST_filters/"+f, skiprows=1)
        #convert to ångström from micron
        raw[:,0] = raw[:,0]*1e4
        filterFuncs.append(np.array([raw[:,0],raw[:,1]]).T)
setFilters()

filtersInit = False
def initFilters(spec):
    #for every filter, interpolate the filter to the spectrum's wavelengths
    global filtersInit
    if filtersInit: return
    filtersInit = True
    global filterFuncs
    filterFuncs_clone = [[] for i in range(len(filterFuncs))]
    for sx in spec[0]:
        for i in range(len(filterFuncs)):
            if filterFuncs[i][0][0] > sx or filterFuncs[i][-1][0] < sx:
                continue
            else:
                index = np.where(filterFuncs[i][:,0] > sx)[0][0]
                leftIndex = index-1
                rightIndex = index
                left = filterFuncs[i][leftIndex]
                right = filterFuncs[i][rightIndex]
                slope = (right[1]-left[1])/(right[0]-left[0])
                cross = left[1] - slope*left[0]
                filterFuncs_clone[i].append([sx, slope*sx+cross])
    for i in range(len(filterFuncs_clone)):
        filterFuncs_clone[i] = np.array(filterFuncs_clone[i])
    filterFuncs = filterFuncs_clone

def performFilter(spec):
    initFilters(spec)
    filterSums = np.zeros(len(filterFuncs), dtype=np.float64)
    filterCounters = np.full(len(filterFuncs), -1, dtype=int)
    for i,x in enumerate(spec[0]):
        for j,count in enumerate(filterCounters):
            if count == -1 and x == filterFuncs[j][0][0]:
                filterCounters[j] += 1
            if count >= 0:
                filterSums[j] += filterFuncs[j][filterCounters[j]][1]*spec[1][i]
                filterCounters[j] += 1
                if filterCounters[j] >= len(filterFuncs[j]):
                    filterCounters[j] = -2
    return filterSums
    #use the james webb filterset
    #https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters

def blackbox(X,threadNum=0):

    # X is a list of input parameters to put into fsps, that thereby generates a spectrum
    # this is then fed through a set of filters

    #unpack X:
    if len(X) != 12: raise ValueError("X must be a list of 12 parameters")
    Z, IMFtemp, zred, const, tau, sf_start, sf_trunc, fburst, tburst, mwr, uvb, tage = X
    if sf_start > tage: sf_start, X[-1] = tage-0.01, tage-0.01
    if const + fburst > 1: const, X[3], fburst, X[7] = const/(const+fburst), const/(const+fburst), fburst/(const+fburst), fburst/(const+fburst)
    IMFtemp, X[1] = round(IMFtemp), round(IMFtemp)

    imfFile = "imf"+str(IMFtemp)+".dat"
    setIMF(imfFile)
    sp = sps.StellarPopulation(
        sfh=4,
        compute_vega_mags=False,
        logzsol=Z,
        imf_type=5,
        dust_type=1,
        zcontinuous=1,
        imf_lower_limit=0.05,
        imf_upper_limit=120,
        add_neb_emission=1,
        zred=zred,
        const=const,
        tau=tau,
        sf_start=sf_start,
        sf_trunc=sf_trunc,
        fburst=fburst,
        tburst=tburst,
        mwr=mwr,
        uvb=uvb
    )
    spec = sp.get_spectrum(tage=tage, peraa=True)
    #spec[0] is wavelength, spec[1] is flux
    Y = performFilter(spec)

    return Y