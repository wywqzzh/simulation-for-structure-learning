#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np


# In[7]:


def tuple_list(l):
    return [tuple(a) for a in l]


# In[8]:


MAP_INFO = pd.read_csv("../Data/mapMsg/map_info.csv")
MAP_INFO = MAP_INFO.loc[MAP_INFO.iswall == 0].reset_index()
data = dict({"pos": tuple_list(MAP_INFO[["Pos1","Pos2"]].values),
             "left": tuple_list(MAP_INFO[["LeftX","LeftY"]].values),
             "right": tuple_list(MAP_INFO[["RightX","RightY"]].values),
             "up": tuple_list(MAP_INFO[["UpX","UpY"]].values),
             "down": tuple_list(MAP_INFO[["DownX","DownY"]].values)})
for key, value in data.items():
    for index,i in enumerate(value):
        if i == (0,0):
            data[key][index] = np.nan
T = pd.DataFrame(data)
T.to_csv("../Data/mapMsg/adjacent_map_fmri.csv")
