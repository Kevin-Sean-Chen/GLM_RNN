#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:28:44 2023

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

###
# Try to generate spiking patterns from Brunel 2000,
# but with latent-driven Poisson neurons, not a LIF-network
###

# %% oscillatory latent
T = 100
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)

freq = .1
latent = .2*np.sin(time*freq*(2*np.pi))
# latent = 2*np.ones(lt)

plt.plot(time,latent)

# %% network setting
N = 30
lk = 10
k_self_spk = -1*np.exp(np.arange(lk)/.1)
# k_self_spk = np.fliplr(k_self_spk[None,:])[0]

ut = np.zeros((N,lt))
yt = np.zeros((N,lt))

C = np.random.randn(N)  # input matrix
# C = np.ones(N)

def NL(x):
    nl = np.exp(x)
    # nl = 1/(1+np.exp(-x))
    return nl
def spiking(nl):
    spk = np.random.poisson(nl)
    # spk = np.random.binomial(1, nl)
    return spk
# %% generate spikes
for tt in range(lk,lt):
    ut[:,tt] = C*latent[tt] + yt[:,tt-lk:tt] @ k_self_spk
    yt[:,tt] = spiking(NL(ut[:,tt]))

# %% plotting
plt.figure()
plt.subplot(211)
plt.imshow(yt,aspect='auto')
plt.subplot(212)
plt.plot(np.sum(yt,0))

# %% brunel_spk

def brunel_spk(phase):
    # setup latent, kernels, and inputs
    if phase=='SR':
        latent = 2*np.ones(lt)
        k_self_spk = -10*np.exp(np.arange(lk)/.1)
        C = np.ones(N)
    elif phase=='AI':
        freq = .1
        latent = .2*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(np.arange(lk)/.1)
        C = np.random.randn(N)
    elif phase=='SIf':        
        freq = .5
        latent = 2*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(np.arange(lk)/.1)
        C = np.ones(N)
    elif phase=='SIs': 
        freq = .1
        latent = 2*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(np.arange(lk)/.1)
        C = np.ones(N)
    
    # simulate spikes
    ut = np.zeros((N,lt))
    yt = np.zeros((N,lt))
    for tt in range(lk,lt):
        ut[:,tt] = C*latent[tt] + yt[:,tt-lk:tt] @ k_self_spk
        yt[:,tt] = spiking(NL(ut[:,tt]))
    
    return yt

def plot_spk(spk):
    plt.figure()
    plt.subplot(211)
    plt.imshow(spk,aspect='auto')
    plt.subplot(212)
    plt.plot(np.sum(spk,0))
    return None

spk = brunel_spk('SR')
plot_spk(spk)

# %%
##############################################################################
# %% inference

