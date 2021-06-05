# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:41:25 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

# %% functions
def energy(x,W,b):
    s = NL(x)
    E = -0.5*s.T @ W @ s + b.T @ s
    return E

def cost(x,y,beta):
    s = NL(x)
    C = beta*(s-y)**2
    return C

def NL(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x#np.tanh(x)

# %%
N = 50
lt = 1000
W = np.random.randn(N,N)
y = np.random.randint(0,2,N)
s = np.random.randint(0,2,N)
x = np.random.randn(N)

equ_step = 10
clamp_step = 10
for tt in range(lt):
    for te in range(equ_step):
        x = W @ s
        s = NL(x)
    for tc in range(clamp_step):
        
        
        
