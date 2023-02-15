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
from scipy.special import logsumexp

import scipy as sp
from scipy.optimize import minimize
from scipy.special import gamma, factorial

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% generate target data from ssm
num_states = 2
obs_dim = 20
input_dim = 1
hmm = ssm.HMM(num_states, obs_dim, input_dim, 
          observations="poisson", #observation_kwargs=dict(C=num_categories),
          transitions="standard")

# %% sample targets
T = 1000
ipt = np.sin(np.arange(0,T)/30)[:,None]
samples = hmm.sample(T=T, input=ipt*0)
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
#tau = np.random.rand(N)*5+dt
tau = 2
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

def lr_unpack(ww,rank):
    b = ww[:N]
    u = ww[N:2*N]
    wl,wr = ww[2*N:(2+rank)*N].reshape(N,rank),ww[(2+rank)*N:(2+rank*2)*N].reshape(rank,N)
    W = wl @ wr
    return b, W, u
    
def negLL(ww, spk, rt,f, dt, lamb=0):
    """
    Negative log-likelihood
    """
    b,W,U = unpack(ww)
#    N = spk.shape[0]
#    lt = spk.shape[1]
#    b,W,U = lr_unpack(ww,rank)
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
#rank = 6
#dd = 2*N*rank+N+N
w_init = np.ones([dd,])*0.1  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, spk_true,rt_true,NL,dt, 10.),w_init,method='L-BFGS-B')#,tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %% adding state information
# if this improves inference, closed-loop latent-state inference would be important
def unpack_state(ww,nstates):
    b = ww[:N]
    u = ww[N:2*N]
    ws = ww[2*N:(2+nstates)*N].reshape(nstates,N)
    W = ww[(2+nstates)*N:].reshape(N,N)
    return b,u,ws,W

def state2onehot(states):
    """
    state vector to onehot encoding, used for state-constrained likelihood
    """
    nstate = np.max(states)+1
    T = len(states)
    onehot = np.zeros((nstate,T))
    for tt in range(T):
        onehot[int(states[tt]),tt] = 1
    return onehot

def negLL_only_states(ws, rt_true, state_onehot):
    """
    classification loss just for the states
    """
    nstates = state_onehot.shape[0]
    ws = ws.reshape(nstates,N)
    lp_states = ws@rt_true #np.exp(ws @ rt_true) #
    # lp_states = lp_states / lp_states.sum(0)[None,:]  # P of class probablity
    lp_states = lp_states - logsumexp(lp_states,0)[None,:]  # logP
    ll = -np.sum(state_onehot * lp_states)
    return ll

def negLL_state(ww, spk, rt, ws, states_onehot, f, dt, lamb=0):
    """
    Poisson log-likelihood plus state-classification loss function, given pre-trained state-readout
    """
    # nstates = states.shape[0]  # state x T
    # b,u,ws,W = unpack_state(ww,nstates)
    b,W,u = unpack(ww)
    temp_f = f(W @ rt + b[:,None] + U[:,None]*ipt.T)
    lp_states = ws @ temp_f  #np.exp(ws @ temp_f)
    # lp_states = lp_states / lp_states.sum(0)[None,:]  # P of class probablity
    lp_states = lp_states - logsumexp(lp_states,0)[None,:]  #logP
    ll = np.sum(spk * np.log(temp_f) - temp_f*dt) - lamb*np.linalg.norm(W)
    state_cost = -np.sum(states_onehot * (lp_states))*lamb
    
    return -ll + state_cost

# %% state inference
onehot = state2onehot(state_true)
dd = N*num_states
ws_init = np.ones([dd,])*0.1
res = sp.optimize.minimize(lambda w: negLL_only_states(w, rt_true, onehot), ws_init, method='L-BFGS-B')
w_map_state = res.x
print(res.fun)
print(res.success)

# %% state-constrained inference
dd = N*N + N + N  #N*num_states
w_init = np.ones([dd,])*0.1  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL_state(w, spk_true,rt_true,w_map_state.reshape(num_states,N),onehot,NL,dt, 10.),\
                           w_init,method='L-BFGS-B')
w_map = res.x
print(res.fun)
print(res.success)

# %% unwrap W matrix full-map
b_rec, W_rec,U_rec = unpack(w_map)
#b_rec, W_rec, U_rec = lr_unpack(w_map,rank)
# b_rec,U_rec,ws_rec,W_rec = unpack_state(w_map,num_states)
spk_rec = np.ones((N,lt))
rt_rec = spk_rec*1
for tt in range(lt-1):
    spk_rec[:,tt] = spiking(NL(W_rec @ (rt_rec[:,tt]) + b_rec*1 + 0*U_rec*ipt[tt])*dt)
    rt_rec[:,tt+1] = rt_rec[:,tt] + dt/tau*(-rt_rec[:,tt] + spk_rec[:,tt]) 

plt.figure()
plt.imshow(spk_rec,aspect='auto')

### higher-levle:
# fit GLM-RNN with ssm output
# should work, but can be impove with latent information
# maybe find a way to do join inference together
# ... capture proabalistic behavior (not driven by input noise!), if possible!

# %% inference back HMM from generated spikes
# Now create a new HMM and fit it to the data with EM
N_iters = 100
hmm_inf = ssm.HMM(num_states, obs_dim, input_dim, 
          observations="poisson", #observation_kwargs=dict(C=num_categories),
          transitions="inputdriven")

# Fit
hmm_lps = hmm_inf.fit(spk_rec.astype(int).T, inputs=ipt, method="em", num_iters=N_iters)

# %%
###
# revise the low-rank constraint method... currently only works for rank-1, somehow
# add in diagonal matrix in the RNN model for generic dicrete time RNN form
###