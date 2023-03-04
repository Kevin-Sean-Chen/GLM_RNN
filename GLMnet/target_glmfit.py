# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 23:21:55 2023

@author: kevin
"""

###
# demo srcipt to simulate population Poisson spikes with complex dynamics, 
# this includes attractors, oscillations, sequences, and chaotic dynamics.
# fit a GLM-RNN to produce this spiking pattern,
# then analyze the inferred parameters for network dynamics.
# The aim is to explore the dynamical repertoire for small noisy RNNs.
###

import matplotlib.pyplot as plt
import ssm
#from ssm.util import one_hot, find_permutation

import autograd.numpy as np
import numpy.random as npr

#from glmrnn.glm_obs_class import GLM_PoissonObservations
from glmrnn.glmrnn import glmrnn
from glmrnn.target_spk import target_spk

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% setup network parameters
N = 10
T = 1000
dt = 0.1
tau = 2
### setup network
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
### setup target spike pattern
d = 1  # latent dimension
my_target = target_spk(N, T, d, my_glmrnn)

# %% produce target spikes
### bistable, oscillation, chaotic, sequence, line_attractor, brunel_spk
targ_spk, targ_latent = my_target.sequence(50)
plt.figure()
plt.imshow(targ_spk, aspect='auto')

# %% test with random input
#input sequence
num_sess = 10 # number of example sessions
input_dim = 1
inpts = np.sin(2*np.pi*np.arange(T)/600)[:,None]*.5 +\
        np.cos(2*np.pi*np.arange(T)/300)[:,None]*1. +\
        .1*npr.randn(T,input_dim)\
        + np.linspace(-2,2,T)[:,None]

inpts = np.repeat(inpts[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

###
# test with 'clock' signal for autonomous system
###
# %% generate training sets
num_sess = 10
true_latents, true_spikes, true_ipt = [], [], []
for sess in range(num_sess):
    true_y, true_z = my_target.sequence(50)  # maybe fix this to pass latent type as string~
    
    true_spikes.append(true_y.T)
#    true_latents.append(true_z[:,None])
    true_latents.append(true_z)
    
#    true_ipt.append(true_z[:,None])
#    true_ipt.append(None)#
#    true_ipt.append(np.zeros(T))   # fix negLL iterations when there is no input vector!
    true_ipt.append(inpts[sess])
    
# %% inference
iid = 1
data = (true_spikes[iid].T, true_ipt[iid])
my_glmrnn.fit_single(data,lamb=0)

# %%
ii = 1
spk,rt = my_glmrnn.forward(true_ipt[ii])
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(true_spikes[ii].T,aspect='auto')
plt.title('true spikes',fontsize=40)
plt.subplot(122)
plt.imshow(spk,aspect='auto')
plt.title('inferred spikes',fontsize=40)

# %% test with batch
#datas = ([true_spikes[0]], [inpts[0]])  # debug this~~~   # might be 'dt'??
datas = (true_spikes, true_ipt)
#my_glmrnn.fit_batch(datas)  # using regression tools
#my_glmrnn.fit_batch_sp(datas)  # this seems to currently work!!...but take too long
my_glmrnn.fit_glm(datas)  # using ssm gradient

# %% test states
datas = (true_spikes, true_ipt, true_latents)
my_glmrnn.fit_glm_states(datas,2)
###