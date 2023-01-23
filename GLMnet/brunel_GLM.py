#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:28:44 2023

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import minimize

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

freq = .5
latent = 1*np.sin(time*freq*(2*np.pi))
latent = 3*np.ones(lt)

plt.plot(time,latent)

# %% network setting
N = 20
lk = 10
k_self_spk = -1*np.exp(-np.arange(lk)/1)
k_self_spk = np.fliplr(k_self_spk[None,:])[0]

ut = np.zeros((N,lt))
yt = np.zeros((N,lt))

# C = np.random.randn(N)  # input matrix
C = np.ones(N)

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
    yt[:,tt] = spiking(NL(ut[:,tt])*dt)

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
        freq = .5
        latent = 4*np.ones(lt)
        k_self_spk = -1*np.exp(-np.arange(lk)/10)
        C = np.ones(N)
    elif phase=='AI':
        freq = .1
        latent = .2*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(-np.arange(lk)/1)
        C = np.random.randn(N)
    elif phase=='SIf':        
        freq = .5
        latent = 1.*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(-np.arange(lk)/1)
        C = np.ones(N)
    elif phase=='SIs': 
        freq = .1
        latent = 1*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(-np.arange(lk)/1)
        C = np.ones(N)
    
    # simulate spikes
    k_self_spk = np.fliplr(k_self_spk[None,:])[0]
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
    plt.xlim([0, spk.shape[1]])
    return None

spk = brunel_spk('SR')
plot_spk(spk)

# %%
##############################################################################
# %% inference
def negLL(ww, spk, f=np.exp, lamb=0):
    N = spk.shape[0]
    lt = spk.shape[1]
    tau = np.abs(ww[:N])
    b = ww[N:2*N]
    W = ww[2*N:].reshape(N,N)
    # evaluate log likelihood and gradient
    rt = np.zeros((N,lt))
    Ks = np.fliplr(np.array([-1*np.exp(-np.arange(lk)/(tt+0.1)) for tt in tau]))
    for tt in range(lk,lt):
        rt[:,tt] = np.sum(spk[:,tt-lk:tt] * Ks , 1)
    ll = np.sum(spk * np.log(f(W @ rt + b[:,None])) - f(W @ rt + b[:,None])*1) \
            - lamb*np.linalg.norm(W)
    return -ll

dd = N*N+N*2  # network, offset, and time-scale
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, spk, NL, 0.),w_init,method='L-BFGS-B',tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %%
def glm_spk(w_map):
    tau = np.abs(w_map[:N])
    b = w_map[N:2*N]
    W = w_map[2*N:].reshape(N,N)
    Ks = np.fliplr(np.array([-1*np.exp(-np.arange(lk)/(tt+0.1)) for tt in tau]))
    spk_rec = np.zeros((N,lt))
    for tt in range(lk,lt):
        ut = np.sum(spk[:,tt-lk:tt] * Ks , 1)
        spk_rec[:,tt] = spiking(NL(W@ut + b))
    return spk_rec

spk_rec = glm_spk(w_map)
plot_spk(spk_rec)