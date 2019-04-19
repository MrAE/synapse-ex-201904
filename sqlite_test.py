#%%
import pickle
import numpy as np
import sqlite3
from sqlite3 import Error


#%%
conn = sqlite3.connect("test.db")


#%%
c = conn.cursor()

#%%
c.execute('''CREATE TABLE synapses
             (id real, data blob)'''
        )


#%%
dat = np.arange(100).reshape(10,10)

#%%
insert = [(1, dat.tobytes())]


#%%
c.executemany('INSERT INTO synapses VALUES (?, ?)', insert)


#%%
c.execute("SELECT * from synapses")

#%%
c.fetchone()

#%%
conn.commit()
conn.close()

#%%

x = pickle.dumps(dat)



#%%

y = pickle.loads(x)

