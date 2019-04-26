import numpy as np
import pickle
from shapely import geometry
import re

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
        self.dilated_areas = geometry.MultiPolygon()
        self.anno_array = np.array(0)
        self.array = np.array(0)
        self.dilated_array = np.array(0)
        self.dilated_bounding_box = set()
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


    def setSynapse(self, Synapse, trans = [32, 32, 32, 32, 1, 1], buff = 10):
        self.id = Synapse.id
        self.areas = shapely.affinity.scale(Synapse.areas, yfact = -1)
        self.dilated_areas = shapely.affinity.scale(Synapse.areas, yfact = -1)
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


    def setDilatedBoundingBox(self, trans, buff = 5):
        ## get centroid from the polygon and buffer
        xy = [float(i) / trans for i in re.findall(r'[0-9]+\.[0-9]+', self.areas.centroid.wkt)]
        z = int(self.anno_bounding_box[5] - self.anno_bounding_box[4])

        self.dilated_bounding_box = {'x': np.asarray([xy[0] - buff[0], xy[0] + buff), 
                                     'y': np.asarray([xy[1] - buff[1], xy[1] + buff[1]), 
                                     'z': np.asarray(np.asarray([z - buff[2], z + buff[2]])}



    def getDilatedBoundingBox(self, rem, collection, experiment, ch):
        di = {
            'rem': rem,
            'ch_rsc': ChannelResource(ch, collection, experiment, 'image',
                                      datatype='uint16'),
            'ch': ch,
            'res': 0,
            'xrng': self.dilated_bounding_box['x'] + np.array([0,1]), # Had to augment to get the last pixel 
            'yrng': self.dilated_bounding_box['y'] + np.array([0,1]), # Had to augment to get the last pixel 
            'zrng': self.dilated_bounding_box['z'], 
        }
        block = di['rem'].get_cutout(di['ch_rsc'], di['res'], di['xrng'], di['yrng'] ,di['zrng'])
        self.channels[str(ch)] = block

        return(0)





