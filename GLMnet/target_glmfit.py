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
T = 200
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
#targ_spk, targ_latent = my_target.bistable()
#targ_spk, targ_latent = my_target.oscillation(50)
#targ_spk, targ_latent = my_target.line_attractor(5)

plt.figure()
plt.imshow(targ_spk, aspect='auto')

# %% test with random input
#input sequence
num_sess = 10 # number of example sessions
input_dim = 1
inpts_ = np.sin(2*np.pi*np.arange(T)/600)[:,None]*.5 +\
        np.cos(2*np.pi*np.arange(T)/300)[:,None]*1. +\
        .1*npr.randn(T,input_dim)\
        + np.linspace(-2,2,T)[:,None]

inpts = np.repeat(inpts_[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

###
# test with 'clock' signal for autonomous system
###
# %% generate training sets
num_sess = 10
true_latents, true_spikes, true_ipt = [], [], []
for sess in range(num_sess):
#    true_y, true_z = my_target.bistable() #
#    true_y, true_z = my_target.sequence(100)  # maybe fix this to pass latent type as string~
#    true_y, true_z = my_target.oscillation(50)
    true_y, true_z = my_target.line_attractor(5)
    
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
ii = 5
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% use rate RNN and Poisson log-likelihood
from glmrnn.rnn_torch import RNN, lowrank_RNN, observed_RNN, RNNTrainer
import torch

# %% setup target
N = 50
T = 200
dt = 0.1
tau = 2
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
d = 1  # latent dimension
my_target = target_spk(N, T, d, my_glmrnn)

num_sess = 100
true_latents, true_spikes, true_ipt = [], [], []
inpts = np.repeat(inpts_[None,:-1,:], num_sess, axis=0)
inpts = list(inpts)
for sess in range(num_sess):
    true_y, true_z = my_target.sequence(50)  # maybe fix this to pass latent type as string~
    true_spikes.append(true_y.T)
    true_latents.append(true_z)
    true_ipt.append(inpts[sess])

# %% tensorize
target_spikes = torch.Tensor(np.array(true_spikes))
target_rates = torch.Tensor(np.transpose(np.array(true_latents),axes=(0, 2, 1)))
inp = torch.Tensor(np.array(true_ipt))

# %% training
inf_net = observed_RNN(1, N, dt, 1) 
masks = torch.ones(num_sess, T+0, N)
trainer = RNNTrainer(inf_net, 'joint', spk_target=target_spikes)
losses = trainer.train(inp, target_rates, masks, n_epochs=100, lr=1e-3, batch_size=5)
### still need to fix poisson ll!

plt.plot(np.arange(len(losses)), losses)
plt.title('Learning curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
