# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 03:22:30 2020

@author: kevin
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
import itertools

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

# %%
def Circuit_entropy(h, h0, J, beta):
    """
    Neural circuit encoding with input potential h, baseline h, coupling J, and inverse temperature beta
    """
    N = len(h)  #number of neurons
    spins = list(itertools.product([-1, 1], repeat=N))  #all binary patterns
    Es = np.zeros(len(spins))
    Ps = np.zeros(len(Es))
    for ii in range(len(spins)):
        vv = np.array(spins[ii])  #enumerating spins
        Es[ii] = -0.5* vv @ J @ vv - (h+h0) @ vv  #driven energy
        Ps[ii] = np.exp(-beta*Es[ii])  #Boltzmann distribution
    Z = sum(Ps)
    Ps = Ps/Z
    H = -np.dot(Ps,np.log(Ps))
    return H
    
def Circuit_encoding(THETA, hs, beta):
    """
    Circuit with theta parameters encoding hs patterns
    """
    N,K = hs.shape  #number of neurons N and patterns K
    h0 = THETA[:N]
    J = THETA[N:]
    J = J.reshape(N,N)
    H_out = Circuit_entropy(np.zeros(N), h0, J, beta)  #output entropy without input
    H_h = np.zeros(K)  #pattern conditional entropy
    for kk in range(K):
        H_h[kk] = Circuit_entropy(hs[:,kk], h0, J, beta)
    MI = H_out - np.mean(H_h)
    return -MI  #max info is minimizing negative information

# %% input and initalization prep
N = 5
K = 5
hs = np.random.randint(0,2,(N,K))
beta = 1
J00 = np.random.randn(N,N)
h00 = np.random.randn(N)
THETA0 = np.concatenate((h00,J00.reshape(-1)))
# %% run optimization
res = minimize(Circuit_encoding, THETA0, args=(hs,beta))

# %% results
xx = res.x
plt.figure()
plt.imshow(xx[N:].reshape(N,N),aspect='auto')

# %% compare with Hopfield
Hop = np.zeros((N,N))
for kk in range(K):
    Hop = Hop + np.outer(hs[:,kk],hs[:,kk])
Hop = Hop/K
plt.figure()
plt.imshow(Hop,aspect='auto')

The_Hop = np.concatenate((xx[:N],Hop.reshape(-1)))
print("Hopfield:", Circuit_encoding(The_Hop, hs, beta))
print("Optimized:", Circuit_encoding(xx, hs, beta))
