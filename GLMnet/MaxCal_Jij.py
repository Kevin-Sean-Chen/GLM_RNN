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
T = 10000
N = 4
gamma = 1  # scaling coupling strength
Jij = np.random.randn(N,N)*gamma/N**0.5
sigma = np.empty((N,T))

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

