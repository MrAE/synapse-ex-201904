
#%%
import intern
import math
import argparse
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
from intern.utils.parallel import block_compute
import configparser
import requests
import numpy as np
from numpy import genfromtxt
import shutil
from IPython.core.debugger import set_trace
import sys
import os
import itertools
from functools import partial
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from intern.utils.parallel import block_compute
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import collections
import json
import pickle
from pathlib import Path
from pprint import pprint
import cv2
from mpl_toolkits.mplot3d import Axes3D
from shapely import geometry
import seaborn
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
        self.array = np.array(0)
        self.anno_bounding_box = set()
        self.bounding_box = set()

    def __str__(self):
        return f"id: {self.id}\nareas: {self.areas}\n"


#%%
fname = "m247514_Take2Site3Annotation_completed_Feb2018_MN_global_synapse_dict.p"

with open(fname, 'rb') as f:
    data = pickle.load(f)


#%%
print(data[j].bounding_box)
print(data[j].bounding_box[0:2])
print(data[j].bounding_box[2:4])
print(data[j].bounding_box[4:6])


#%%
def toNumpyMask():

    return(0)




#%%
## For getting masked bounding box around centroid of annotation
ch = CHAN_NAMES[-1]
di = [{
      'rem': rem,
      'ch_rsc': ChannelResource(ch, collection, experiment, 'image',
                                datatype='uint16'),
      'ch': ch,
      'res': 0,
      'xrng': [int(data[j].bounding_box[0:2][0]/32), int(data[j].bounding_box[0:2][1]/32)],
      'yrng': [int(data[j].bounding_box[2:4][0]/32), int(data[j].bounding_box[2:4][1]/32)],
      'zrng': [int(data[j].bounding_box[4:6][0]), int(data[j].bounding_box[4:6][1])],
     } for j in list([551])]


#%%
dim = di[0]
block = dim['rem'].get_cutout(dim['ch_rsc'], dim['res'], dim['xrng'], dim['yrng'] ,dim['zrng'])
#
seaborn.heatmap(block[0])
seaborn.heatmap(block[1])
seaborn.heatmap(block[2])


#%%
ch = CHAN_NAMES[2]
testDat = [{
      'rem': rem,
      'ch_rsc': ChannelResource(ch, collection, experiment, 'image',
                                datatype='uint16'),
      'ch': ch,
      'res': 0,
      'xrng': [2233, 2242], 
      'yrng': [2342, 2349], 
      'zrng': [20, 23], 
     } for j in list([551])]

dim = testDat[0]
block = dim['rem'].get_cutout(dim['ch_rsc'], dim['res'], dim['xrng'], dim['yrng'] ,dim['zrng'])
#



#%%
fig, axs = plt.subplots(ncols=3)
seaborn.heatmap(block[0], ax = axs[0])
seaborn.heatmap(block[1], ax = axs[1])
seaborn.heatmap(block[2], ax = axs[2])

#%%
CHAN_NAMES














#%%
j = 551
d = data[j]

fig, axs = plt.subplots(nrows = d.array.shape[2])
for i in range(d.array.shape[2]):
    a = d.array[:, :, i]
    seaborn.heatmap(np.transpose(a), ax = axs[i])

#%%
class AugSynapse:
    def __init__(self, id):
        self.id = id
        self.areas = geometry.MultiPolygon()
        self.array = np.array(0)
        self.mask = np.array(0)
        self.em_bounding_box = set()
        self.at_bounding_box = set()

    def __str__(self):
        return f"id: {self.id}\nareas: {self.areas}\n"

    def em2at(resTrans):
        self.at_bounding_box = \
            [int(self.em_bounding_box[i]/ resTrans[i]) for i in range(len(resTrans))]



#%%
w = AugSynapse(551)
w.areas = data[551].areas
w.mask = np.transpose(data[551].array)
w.em_bounding_box = data[551].bounding_box


#%%
seaborn.heatmap(w.mask[2, ::])

#%%
w.mask.shape
ch = CHAN_NAMES[-1]
testDat = [{
      'rem': rem,
      'ch_rsc': ChannelResource(ch, collection, experiment, 'image',
                                datatype='uint16'),
      'ch': ch,
      'res': 0,
      'xrng': [2233, 2242], 
      'yrng': [2342, 2349], 
      'zrng': [20, 23], 
     } for j in list([551])]

dim = testDat[0]
block = dim['rem'].get_cutout(dim['ch_rsc'], dim['res'], dim['xrng'], dim['yrng'] ,dim['zrng'])
#

#%%
w.em_bounding_box



#%%
resTrans = [32, 32, 32, 32, 1, 1]
[int(w.em_bounding_box[i]/ resTrans[i]) for i in range(len(resTrans))]

