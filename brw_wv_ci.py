# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 21:09:54 2022

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %%
def BRW(brw,dc):
    """
    Biased random walk modulated by a 0-1 factor governing turning porbabiliy (run length)
    """
    temp = np.random.rand()
    if dc>0:
        if temp<0.5+brw/2:
            rot = 0
        else:
            rot = 1
    elif dc<=0:
        if temp<0.5-brw/2:
            rot = 0
        else:
            rot = 1
    return rot  ### 1 for turn 0 for run

def WV(wv,dc):
    """
    Weathervaning strategy modulating turning direction (given a turn)
    """
    temp = np.random.rand()
    if dc>0:
        if temp<0.5+wv:
            turn = +1
        else:
            turn = -1
    elif dc<=0:
        if temp<0.5-wv:
            turn = -1
        else:
            turn = +1
    return turn  ###+ or - sign for 1D direction

# %% paramters
N = 50
T = 1000
brw = 0.1
wv = 0.1

# %%
tracks = np.zeros((N,T))
for nn in range(N):
    track_i = np.zeros(T)
    track_i[1] = -1#np.random.randn()
    for tt in range(1,T-1):
        dc = track_i[tt] - track_i[tt-1]
        rot = BRW(brw, dc)
        if rot == 0:
            track_i[tt+1] = track_i[tt] + 1*np.sign(dc)  #run in same direction
        elif rot == 1:
            turn = WV(wv, dc)
            track_i[tt+1] = track_i[tt] + 1*turn  #turning according to a sign
    
    tracks[nn,:] = track_i
    
# %%
plt.figure()
plt.plot(tracks.T)
print(np.mean(tracks[:,-1])/T)

# %%
bbs = np.arange(0,0.4,0.05)
wws = np.arange(0,0.4,0.05)
cis = np.zeros((len(bbs),len(wws)))
for ii in range(len(bbs)):
    for jj in range(len(wws)):
        tracks = np.zeros((N,T))
        brw, wv = bbs[ii], wws[jj]
        for nn in range(N):
            track_i = np.zeros(T)
            track_i[1] = -1#np.random.randn()
            for tt in range(1,T-1):
                dc = track_i[tt] - track_i[tt-1]
                rot = BRW(brw, dc)
                if rot == 0:
                    track_i[tt+1] = track_i[tt] + 1*np.sign(dc)  #run in same direction
                elif rot == 1:
                    turn = WV(wv, dc)
                    track_i[tt+1] = track_i[tt] + 1*turn  #turning according to a sign
            
            tracks[nn,:] = track_i
        cis[ii,jj] = np.mean(tracks[:,-1])/T
    print(ii)
    
# %%
plt.figure()
plt.imshow(cis.T,interpolation='gaussian',origin='lower', \
           cmap = 'viridis', extent=[np.min(bbs),np.max(bbs),np.min(wws),np.max(wws)])
plt.colorbar()