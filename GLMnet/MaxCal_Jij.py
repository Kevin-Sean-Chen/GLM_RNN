# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:36:12 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% Discrete-time binary neural network model
def spiking(p_spk, spk_past):
    """
    I: firing probability vector and past spike
    O: binary spiking vector
    """
    N = len(p_spk)
    rand = np.random.rand(N)
    spk = np.zeros(N)
    spk[p_spk>rand] = 1  # probablistic firing
    spk[spk_past==1] = 0  # refractoriness
    return spk

def nonlinearity(x):
    """
    I: current vector
    O: firing probability vector
    """
    f = 1/(1+np.exp(-x))  # sigmoid nonlinearity
    return f

def current(Jij, spk):
    baseline = 0
    I = Jij @ spk + baseline  # coupling and baseline
    return I

# %% initializations
T = 100000
N = 4
gamma = 1  # scaling coupling strength
Jij = np.random.randn(N,N)*gamma/N**0.5  # coupling matrix
sigma = np.zeros((N,T))

# %% Neural dynamics
for tt in range(0,T-1):
    p_spk = nonlinearity( current(Jij, sigma[:,tt]) )
    sigma[:,tt+1] = spiking(p_spk, sigma[:,tt])

plt.figure()
plt.imshow(sigma,aspect='auto',cmap='Greys',  interpolation='none')

# %% Inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% functions
eps = 10**-10
def inv_f(x):
    invf = -np.log((1 / (x + eps)) - 1)
    return invf

def J_inf(x):
    
    return Jij

# %% counting through time
Jinf = Jij*0  #inferred Jij

for ii in range(N):
    for jj in range(N):
        temp = sigma[[ii,jj],:]  # selectec i,j pair
        if ii is not jj:
            ### counting inference
            pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]
            n10 = len(pos10)
            pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
            pos10_01 = np.intersect1d(pos10+1, pos01)
            n10_01 = len(pos10_01)
            p_1001 = n10_01 / n10
            
            pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
            n00 = len(pos00)
            pos00_01 = np.intersect1d(pos00+1, pos01)
            n00_01 = len(pos00_01)
            p_0001 = n00_01 / n00
            
            Jinf[ii,jj] = inv_f(p_1001) - inv_f(p_0001)
plt.figure()
plt.plot(Jij, Jinf, 'ko')
plt.xlabel('true Jij', fontsize=30)
plt.ylabel('inferred Jij', fontsize=30)