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
# then fit GLM-network to the target spikes
###

# %% oscillatory latent
T = 100
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)

freq = .5
latent = 1*np.sin(time*freq*(2*np.pi))
# latent = 3*np.ones(lt)

plt.plot(time,latent)

# %% network setting
N = 10
lk = 10
k_self_spk = -1*np.exp(-np.arange(lk)/1)
k_self_spk = np.fliplr(k_self_spk[None,:])[0]

ut = np.zeros((N,lt))
yt = np.zeros((N,lt))

# C = np.random.randn(N)  # input matrix
C = np.ones(N)

def NL(x):
    """
    Spiking nonlinearity
    """
    # nl = np.exp(x)
    nl = 1/(1+np.exp(-x))
    # nl = np.log(1+np.exp(x))
    return nl
def spiking(nl):
    """
    Spiking process
    """
    ### exp-Poisson
    # spk = np.random.poisson(nl)
    
    ### sigmoid-Bernoulli
    # nl = 1-np.exp(-nl)
    spk = np.random.binomial(1, nl)
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
    """
    Latent-driven Poisson spiking patterns to mimic Bruenl 2000 firing patterns,
    with phases SR, AI, SIf, and SIs
    """
    # setup latent, kernels, and inputs
    if phase=='SR':
        latent = 3*np.ones(lt)
        k_self_spk = -20*np.exp(-np.arange(lk)/20)
        C = np.ones(N)
    elif phase=='AI':
        freq = .1
        latent = .2*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(-np.arange(lk)/1)
        C = np.random.randn(N)
    elif phase=='SIf':        
        freq = .7
        latent = 2.*np.sin(time*freq*(2*np.pi))
        k_self_spk = -1*np.exp(-np.arange(lk)/1)
        C = np.ones(N)
    elif phase=='SIs': 
        freq = .1
        latent = 2*np.sin(time*freq*(2*np.pi))
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
    plt.imshow(spk[:,lk:],aspect='auto')
    plt.ylabel('neurons', fontsize=25)
    plt.xticks([])
    plt.subplot(212)
    plt.plot(np.sum(spk[:,lk:],0))
    plt.xlim([0, spk.shape[1]])
    plt.ylabel('pop. rate', fontsize=25)
    plt.xlabel('time', fontsize=25)
    return None

spk = brunel_spk('AI')
plot_spk(spk)

# %% true GLM (as positive control for inference)
def glm_spk(w_map):
    tau = np.abs(w_map[0]) #np.abs(w_map[:N])
    b = w_map[1:N+1]
    W = w_map[N+1:].reshape(N,N)
    # Ks = np.fliplr(np.array([-1*np.exp(-np.arange(lk)/(tt+0.1)) for tt in tau]))
    ks = -1*np.exp(-np.arange(lk)/tau)
    ks = np.fliplr(ks[None,:])[0]
    spk_rec = np.zeros((N,lt))
    for tt in range(lk,lt):
        ut = spk_rec[:,tt-lk:tt] @ ks
        spk_rec[:,tt] = spiking(NL(W@ut + b))
    return spk_rec

t_gt, b_gt, ww_gt = np.ones(1), np.ones(N)*0.1, np.random.randn(N*N)*0.7
w_gt = np.concatenate((t_gt, b_gt, ww_gt))
spk_gt = glm_spk(w_gt)
plot_spk(spk_gt)

# %%
##############################################################################
# %% inference
def unpack(ww):
    """
    Vector of weights to the parameters in GLM,
    used to unpack for optimization
    """
    # tau = np.abs(ww[:N])
    # b = ww[N:2*N]
    # W = ww[2*N:].reshape(N,N)
    tau = ww[0]
    b = ww[1:N+1]
    W = ww[N+1:].reshape(N,N)
    return tau, b, W

def negLL(ww, spk, f=np.exp, lamb=0):
    """
    Negative log-likelihood of the 
    """
    N = spk.shape[0]
    lt = spk.shape[1]
    tau,b,W = unpack(ww)
    # evaluate log likelihood and gradient
    rt = np.zeros((N,lt))
    ks = -1*np.exp(-np.arange(lk)/tau)
    ks = np.fliplr(ks[None,:])[0]
    for tt in range(lk,lt):
        rt[:,tt] = spk[:,tt-lk:tt] @ ks
    ### Poisson log-likelihood
    ll = np.sum(spk * np.log(f(W @ rt + b[:,None])) - f(W @ rt + b[:,None])*1) \
            - lamb*np.linalg.norm(W)
    ### Bernoulli log-likelihood
    # ll = np.nansum(spk * np.log(f(W @ rt + b[:,None])) - (1-spk)*np.log(1-f(W @ rt + b[:,None])))
    return -ll

dd = N*N + N + 1  # network, offset, and time-scale
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
tb, bb, wb = (0.1,None), (None,None), (None,None)
def rep_bnd(bnd, rep):
    bnds = []
    for i in range(rep):
        bnds.append(bnd)
    return bnds
tbs, bbs, wbs, = rep_bnd(tb, 1), rep_bnd(bb, N), rep_bnd(wb, N*N)
bnds = sum([tbs, bbs, wbs], [])
res = sp.optimize.minimize(lambda w: negLL(w, spk, NL, 0.),w_init,method='L-BFGS-B',bounds=bnds)#,tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %% reconstruct activity from inferred weights
spk_rec = glm_spk(w_map)
plot_spk(spk_rec)

