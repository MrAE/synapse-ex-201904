
#%%
import synapseClasses
from pathlib import Path
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

    p3 = Path("outputs_m247514_Take2Site3")    
    p4 = Path("outputs_m247514_Take2Site4")    

    f3 = sorted(list(p3.glob("*.pickle")))
    f4 = sorted(list(p4.glob("*.pickle")))

     
    site3 = {}
    site4 = {}

    for f3i in f3:
        with open(f3i, "rb") as ff3:
            si = pickle.load(ff3)
            site3[str(si.id)] = np.mean(si.array)
        

    for f4i in f4:
        with open(f4i, "rb") as ff4:
            si = pickle.load(ff4)
            site4[str(si.id)] = np.mean(si.array)
        


    
    print(site3)
