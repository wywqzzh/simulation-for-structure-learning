#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx


# In[2]:


T = pd.read_pickle("../Data/mapMsg/adjacent_map_fmri.pickle")
G = nx.Graph()
G.add_nodes_from(T.pos)
for i in range(0,T.shape[0]):
    k = T.pos[i]
    G.add_edges_from(([(k,t) for t in T.iloc[i,1:5].values if t is not np.nan]))
def get_relative_dir(pos1,pos2):
    res = tuple(map(lambda i, j: j - i, Source, Target))
    if res[0] == 0 and res[0] > 0:
        return ['down']
    if res[0] == 0 and res[0] < 0:
        return ['up']
    if res[0] > 0 and res[1] == 0:
        return ['right']
    if res[0] < 0 and res[1] == 0:
        return ['left']
    if res[0] > 0 and res[1] > 0:
        return ['right', 'down']
    if res[0] > 0 and res[1] < 0:
        return ['right', 'up']
    if res[0] < 0 and res[1] > 0:
        return ['left', 'down']
    if res[0] < 0 and res[1] < 0:
        return ['left', 'up']


# In[5]:


Tr = {"pos1":[],"pos2":[],"dis":[],"path":[],"relative_dir":[]}
for Source in T.pos:
    for Target in T.pos:
        if Source == Target:
            continue
        Tr['pos1'].append(Source)
        Tr['pos2'].append(Target)
        Tr['dis'].append(nx.shortest_path_length(G,Source,Target))
        Tr['path'].append([x for x in nx.all_shortest_paths(G,Source,Target)])
        Tr['relative_dir'].append(get_relative_dir(Source,Target))
pos1 = Tr.get("pos1")
pos2 = Tr.get("pos2")
dis = Tr.get("dis")
path = Tr.get("path")
relative_dir = Tr.get("relative_dir")
df = pd.DataFrame({"pos1":pos1,"pos2":pos2,"dis":dis,"path":path,"relative_dir":relative_dir})
df.to_csv("../Data/mapMsg/dij_distance_map_fmri.csv")

