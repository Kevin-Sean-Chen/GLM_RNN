#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:55:20 2023

@author: kschen
"""
import numpy as np
import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation

import scipy as sp
from scipy.optimize import minimize
from scipy.special import gamma, factorial

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% generate target data from ssm
num_states = 3
obs_dim = 10
input_dim = 1
hmm = ssm.HMM(num_states, obs_dim, input_dim, 
          observations="poisson", #observation_kwargs=dict(C=num_categories),
          transitions="inputdriven")

# %%
T = 1000
samples = hmm.sample(T=T)
state_true = samples[0]
spk_true = samples[1]

plt.figure()
plt.subplot(211)
plt.plot(state_true)
plt.subplot(212)
plt.imshow(spk_true.T, aspect='auto')

# %% GLM inference here
def negLL(ww, spk, rt, dt, f=np.exp, lamb=0):
    N = spk.shape[0]
    b = ww[:N]
    W = ww[N:].reshape(N,N)
#    W = ww.reshape(N,N)
    # poisson log likelihood
    ll = np.sum(spk * np.log(f(W @ phi(rt) + b[:,None])) - f(W @ phi(rt) + b[:,None])*dt) \
            - lamb*np.linalg.norm(W) \
            - lamb*(W.T @ W).sum()
    return -ll

dd = N*N+N
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, spk,rt,dt,NL, 0.),w_init,method='L-BFGS-B')#,tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %% unwrap W matrix full-map
brec = w_map[:N]
Wrec = w_map[N:].reshape(N,N)*1.

# low-rank constrained
#brec = w_lr[:N]
#mv, nv= w_lr[N:2*N], w_lr[2*N:]
#Wrec = np.outer(mv,nv)
#brec = 0

# %% simulated given inferred parameters
spk_rec = np.zeros((N,lt))
rt_rec = spk_rec*0
for tt in range(lt-1):
    spk_rec[:,tt] = spiking(NL(Wrec @ phi(rt_rec[:,tt]) + brec*1)*dt)
    rt_rec[:,tt+1] = rt_rec[:,tt] + dt/tau_r*(-rt_rec[:,tt] + spk_rec[:,tt]) 

plt.figure()
plt.imshow(rt_rec,aspect='auto')

### higher-levle
# fit GLM-RNN with ssm output
# should work, but can be impove with latent information
# maybe find a way to do join inference together