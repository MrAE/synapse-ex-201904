
#%%
import synapseClasses
import intern
import math
import argparse
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
from intern.utils.parallel import block_compute
import configparser
import numpy as np
import pickle
from shapely import geometry
import shapely
import seaborn
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
#import requests
#from numpy import genfromtxt
#import shutil
#from IPython.core.debugger import set_trace
#import sys
#import os
#import itertools
#from functools import partial
#from intern.utils.parallel import block_compute
#%matplotlib inline
#import matplotlib
#import matplotlib.pyplot as plt
#import collections
#import json
#import pickle
#from pathlib import Path
#from pprint import pprint
#import cv2
#from mpl_toolkits.mplot3d import Axes3D
#from cloudvolume import CloudVolume
#from tqdm import tqdm

CONFIG_FILE = 'config.ini'

config = configparser.ConfigParser()
config.read(CONFIG_FILE)
TOKEN = config['Default']['token']
boss_url = ''.join( ( config['Default']['protocol'],'://',config['Default']['host'],'/v1/' ) )
collection = config['Default']['boss_collection']
experiment = config['Default']['boss_experiment']
#print(boss_url)

#%%
rem = BossRemote(CONFIG_FILE)

CHAN_NAMES = rem.list_channels(collection, experiment) 


cf = CoordinateFrameResource(str(collection + '_' + experiment))
cfr = rem.get_project(cf)

ex = {'x': cfr.x_stop, 'y': cfr.y_stop, 'z': cfr.z_stop}


#%%
def getAnnoData(di):
    data = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'],
          di['yrng'] ,di['zrng'])
    out = np.multiply(data, di['mask'])
    return(out)

def getMaskData(di):
    data = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'],
          di['yrng'] ,di['zrng'])
    out = np.multiply(data, di['mask'])
    return(out)

def getMaskDataT(di):
    data = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'],
          di['yrng'] ,di['zrng'])
    out = np.multiply(data, di['mask']).astype(data.dtype)
    return(out)

def getCube(di):
    data = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'],
          di['yrng'] ,di['zrng'])
    return(data)

def getWCube(di):
    data = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'],
          di['yrng'] ,di['zrng'])
    y = np.multiply(data, di['w']) 
    return(y)

def getCentroid(box):
    m = np.asarray(box == True) 
    avg = np.int(np.round(np.mean(m,1)))
    return(avg)

def weightCubes(cubes, w):
    c = np.float32(cubes)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            tmp = np.multiply(c[i,j,::], w)
            c[i,j,::] = tmp
    return(c)

def distMat2(bf):
    A = np.reshape(np.array([np.sqrt((i-(bf+1))**2 + (j-(bf+1))**2)
    for i in range(1, 2*bf+2)
    for j in range(1, 2*bf+2)]), (2*bf+1, 2*bf+1))
    A[np.where(A == 0)] = 1
    return(A)

def distMat3(bf):
    A = np.reshape(
        np.array([np.sqrt((i-(bf+1))**2 + (j-(bf+1))**2 + (k-(bf+1))**2)\
        for i in range(1, 2*bf+2)\
        for j in range(1, 2*bf+2)\
        for k in range(1, 2*bf+2)]),(2*bf+1, 2*bf+1, 2*bf+1))
    #A[np.where(A == 0)] = 1
    return(A)

def F0(cubes):
    print(cubes.shape)
    f0 = [[np.sum(cubes[i,j,:,:,:]) for j in range(cubes.shape[1])] for i in range(cubes.shape[0])]
    #f0 = np.sum(cube)
    return(f0)


#%%
class Synapse:
    def __init__(self, id):
        self.id = id
        self.areas = geometry.MultiPolygon()
        self.array = np.array(0)
        self.bounding_box = set()

    def __str__(self):
        return f"id: {self.id}\nareas: {self.areas}\n"

#%%

class AugmentedSynapse:
    def __init__(self, id):
        self.id = id
        self.areas = geometry.MultiPolygon()
        self.anno_array = np.array(0)
        self.array = np.array(0)
        self.anno_bounding_box = set()
        self.bounding_box = set()
        self.channels = {}


    def save(self):
        return (self.__class__, self.__dict__)


    def load(cls, attributes):
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return(obj)

    def __str__(self):
        return f"id: {self.id}\nareas: {self.areas}\n"


    def setSynapse(self, Synapse, trans = [32, 32, 32, 32, 1, 1]):
        self.id = Synapse.id
        self.areas = shapely.affinity.scale(Synapse.areas, yfact = -1)
        self.anno_bounding_box = Synapse.bounding_box
        self.bounding_box = [int (Synapse.bounding_box[i] / trans[i]) for i in range(len(trans))]
        self.anno_array = np.transpose(Synapse.array)
        
    
    def getChannel(self, rem, collection, experiment, ch):
        di = {
            'rem': rem,
            'ch_rsc': ChannelResource(ch, collection, experiment, 'image',
                                      datatype='uint16'),
            'ch': ch,
            'res': 0,
            'xrng': self.bounding_box[0:2] + np.array([0,1]), # Had to augment to get the last pixel 
            'yrng': self.bounding_box[2:4] + np.array([0,1]), # Had to augment to get the last pixel 
            'zrng': self.bounding_box[4:6], 
        }
        block = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'], di['yrng'] ,di['zrng'])
        self.channels[str(ch)] = block

#%%
def drone(args):
    fname, rem, collection, experiment, ch = args
    #print("Loading object:")
    with open(fname, "rb") as f:
        obj = pickle.load(f)

    #print("retrieving BOSS data:")
    obj.getChannel(rem, collection, experiment, ch)

    #print("saving object " + str(obj.id) + ":\n")
    with open(fname, "wb") as fout:
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
#%%
    #fname = "m247514_Take2Site3Annotation_completed_Feb2018_MN_global_synapse_dict.p" 
    fname = "m247514_Take2Site4Annotation_MN_Take2Site4global_synapse_dict.p"

    with open(fname, 'rb') as f:
        data = pickle.load(f)



#%%
    W = {i : AugmentedSynapse(i) for i in data}

    for w in W:
        W[w].setSynapse(data[W[w].id])

    names = {i: "outputs/id" + str(i).zfill(3) + ".pickle" for i in W}


#%%
    for i in names:
        with open(names[i], "wb") as f:
            pickle.dump(W[i], f, pickle.HIGHEST_PROTOCOL)


#%%
    for ch in CHAN_NAMES:
        Args = [[[names[i], rem, collection, experiment, ch]] for i in names]
        print(ch)
        with ThreadPool(12) as thb:
            thb.starmap(drone, Args)




#%%
