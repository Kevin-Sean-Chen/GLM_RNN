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
obs_dim = 20
input_dim = 1
hmm = ssm.HMM(num_states, obs_dim, input_dim, 
          observations="poisson", #observation_kwargs=dict(C=num_categories),
          transitions="inputdriven")

# %% sample targets
T = 1000
ipt = np.sin(np.arange(0,T)/30)[:,None]
samples = hmm.sample(T=T, input=ipt)
state_true = samples[0]
spk_true = samples[1].T

plt.figure()
plt.subplot(211)
plt.plot(state_true)
plt.subplot(212)
plt.imshow(spk_true, aspect='auto')

# %% tranistion to GLM network
N = obs_dim*1  # neurons
lt = T*1  # time
dt = 0.1  # time step
lk = 20  # kernel length
tau = np.random.rand(N)*5+dt
#tau = 2
U = np.random.randn(N)*.1 #input vector

### process for rate vectors
rt_true = np.zeros((N,lt))
for tt in range(lt-1):
     rt_true[:,tt+1] = rt_true[:,tt] + dt/tau*(-rt_true[:,tt] + spk_true[:,tt])

# %% GLM inference
def NL(x):
#    nl = np.exp(x)
    nl = np.log(1+np.exp(x))
    return nl

def spiking(x):
    spk = np.random.poisson(x)
    return spk

def unpack(ww):
    """
    Vector of weights to the parameters in GLM,
    used to unpack for optimization
    """
    # tau = np.abs(ww[:N])
    # b = ww[N:2*N]
    # W = ww[2*N:].reshape(N,N)
#    tau = np.abs(ww[0])+dt # for temporal stability
#    b = ww[1:N+1]
#    W = ww[N+1:].reshape(N,N)
    b = ww[:N]
    u = ww[N:2*N]
    W = ww[2*N:].reshape(N,N)
    return b, W, u

def lr_unpack(ww):
    b = ww[:N]
    wl,wr = ww[N:3*N].reshape(N,2),ww[3*N:5*N].reshape(2,N)
    W = wl @ wr
    return b, W
    
def negLL(ww, spk, rt,f, dt, lamb=0):
    """
    Negative log-likelihood
    """
    b,W,U = unpack(ww)
#    N = spk.shape[0]
#    lt = spk.shape[1]
#    b,W = lr_unpack(ww)
    # evaluate log likelihood and gradient
#    rt = np.zeros((N,lt))
#    ks = 1*np.exp(-np.arange(lk)/tau)
#    ks = np.fliplr(ks[None,:])[0]
#    for tt in range(lk,lt):
#        rt[:,tt] = spk[:,tt-lk:tt] @ ks
    
#    rt = np.zeros((N,lt))
#    for tt in range(lt-1):
#         rt[:,tt+1] = rt[:,tt] + dt/tau*(-rt[:,tt] + spk[:,tt])
    
    ### Poisson log-likelihood
    ll = np.sum(spk * np.log(f(W @ rt + b[:,None] + U[:,None]*ipt.T)) \
                - f(W @ rt + b[:,None] + U[:,None]*ipt.T)*dt) \
            - lamb*np.linalg.norm(W) #\
#            - lamb*np.sum(f(W @ rt + b[:,None])[:,:-1]*dt-rt_true[:,1:])**2
            ### add catigorical loss function here, for discrete firing patterns.
    return -ll

dd = N*N+N+N
w_init = np.ones([dd,])*0.1  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, spk_true,rt_true,NL,dt, 10.),w_init,method='L-BFGS-B')#,tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %% unwrap W matrix full-map
b_rec, W_rec,U_rec = unpack(w_map)
spk_rec = np.zeros((N,lt))
rt_rec = spk_rec*0
for tt in range(lt-1):
    spk_rec[:,tt] = spiking(NL(W_rec @ (rt_rec[:,tt]) + b_rec*1 + U_rec*ipt[tt])*dt)
    rt_rec[:,tt+1] = rt_rec[:,tt] + dt/tau*(-rt_rec[:,tt] + spk_rec[:,tt]) 

plt.figure()
plt.imshow(spk_rec,aspect='auto')

### higher-levle:
# fit GLM-RNN with ssm output
# should work, but can be impove with latent information
# maybe find a way to do join inference together
# ... capture proabalistic behavior (not driven by input noise!), if possible!

# %% 