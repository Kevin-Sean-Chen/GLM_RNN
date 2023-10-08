# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 01:21:27 2023

@author: kevin
"""

###
# demo srcipt to simulate GLM-HMM with Gaussian observations, 
# fit a GLM-RNN to produce this stochastic transitioning pattern,
# then do inference to recover RNN parameters, with state constraints
###

import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation

import autograd.numpy as np
import numpy.random as npr

from rnn_torch import RNN, RNN_state, observed_RNN, RNNTrainer, state2onehot
import torch

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% make SSM target data
###############################################################################
# %%
num_states = 2        # number of discrete states           K
obs_dim = 50          # number of observed dimensions       D
input_dim = 1         # input dimensions                    M

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="gaussian", transitions="inputdriven")  
# observation class would be replaced to GLM_Pousson
# transitions: sticky , standard , inputdriven
# possible to be both driven!

# input driving conditions
driven_emission = 0
driven_state = 0
drive_logic = np.min([driven_emission+driven_state,1])

# %%  replace from here for now
if driven_emission==1:
    true_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim)  # implement Gaussian input emission
    true_glmhmm.observations.Wk *= 3.
    print(true_glmhmm.observations.Wk.shape)
if driven_state==1 :
    true_glmhmm.transitions.Ws *= 3.
    print(true_glmhmm.transitions.Ws)
else:
    print(true_glmhmm.transitions.transition_matrix)

#true_glmhmm.observations.mus += np.array([10,-10])[:,None]  # better scaling
true_glmhmm.observations.mus *= 5

# %%
#input sequence
num_sess = 50 # number of example sessions
time_len = 500 # time duration of simulation
inpts = np.ones((num_sess, time_len, input_dim)) # initialize inpts array
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]

### discrete input
#inpts[:,:,0] = np.random.choice(stim_vals, (num_sess, time_len)) # generate random sequence of stimuli
#inpts[:,:,1] = np.random.choice(stim_vals, (num_sess, time_len)) 
### random input
#inpts = np.random.randn(num_sess, num_trials_per_sess, input_dim)
### noisy sine input
inpts = np.sin(2*np.pi*np.arange(time_len)/600)[:,None]*.5*0 +\
        np.cos(2*np.pi*np.arange(time_len)/200)[:,None]*1.*0 +\
        .1*npr.randn(time_len,input_dim)*0 + \
        np.arange(0,time_len,1)[:,None]/time_len

inpts = np.repeat(inpts[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

# different for each session
inpts = []
for ii in range(num_sess):
#    inpt_ = np.arange(0,time_len,1)[:,None]/time_len*10. + .1*npr.randn(time_len,input_dim)*1 # noisy ramp
    inpt_ = np.sin(2*np.pi*np.arange(time_len)/600)[:,None]*.1 +\
        np.cos(2*np.pi*np.arange(time_len)/200)[:,None]*.1 +\
        .1*npr.randn(time_len,input_dim)*.5
    inpts.append(inpt_)

# %% # Generate a sequence of latents and choices for each session
true_latents, true_spikes = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(time_len, input=inpts[sess]*drive_logic)  #changed hmm.py line206!
    true_latents.append(true_z)
    true_spikes.append(true_y)

# %% train with RNN
###############################################################################
# %%
net_size = obs_dim*2
deltaT = .1
#my_net = observed_RNN(input_dim, net_size, deltaT, 1.)  # input_dim, N, r, output_dim, dt, init_std=1.
my_net = RNN_state(input_dim, net_size, obs_dim, num_states, deltaT, 1.) 
#my_net = RNN(input_dim, net_size, obs_dim, deltaT, 1.)
# input_dim, N, output_dim, state_dim, dt, init_std=1  # fully observed so output==network, for now?

my_net = my_net.float()
masks = torch.ones(num_sess, time_len+0, obs_dim)  # trial x T x N
target_spk = torch.Tensor(np.array(true_spikes))
max_abs_value = torch.max(torch.abs(target_spk))
target_spk = target_spk / max_abs_value

target_sts = state2onehot(np.array(true_latents))
target_ipt = torch.Tensor(np.array(inpts))

# %%
#trainer = RNNTrainer(my_net, 'MSE', spk_target=target_spk, st_target=target_sts)
trainer = RNNTrainer(my_net, 'state', spk_target=target_spk, st_target=target_sts)
trainer.alpha = .5
losses = trainer.train(target_ipt, target_spk, masks, n_epochs=50, lr=1e-2, batch_size=5)
### still need to fix poisson ll!

plt.plot(np.arange(len(losses)), losses)
plt.title('Learning curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% trained results
# %% generative with rate
trid = 10
#_, out_rt = my_net.forward(target_ipt)
_, out_rt, out_st = my_net.forward(target_ipt)
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.title('true activity',fontsize=40)
plt.imshow(target_spk[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')
plt.subplot(122)
plt.title('inferred activity',fontsize=40)
plt.imshow(out_rt[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')

# %%
#trid = 0
#_, out_rt = my_net.forward(target_ipt)
#_, out_rt, out_st = my_net.forward(target_ipt)
plt.figure()
plt.figure(figsize=(15,10))
plt.subplot(211)
plt.title('true state',fontsize=40)
plt.imshow(target_sts[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')
plt.subplot(212)
plt.title('inferred state',fontsize=40)
plt.imshow(out_st[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')
###
# CHECKKKK state transition function... time and state dimensions are correct?
# check if noise are properly learned?
# does state-constaint work? -> seems to!?

# what is a better noise model? can it generalize to other exp family variables?
# ... turn to a stochastic RNN model with only state and behavioral targets (not neural activity!)
# step-wise training? --> state first, then emission
# ... try variational inference!?

### training methods
# feedback the state signal --> similar to chaotic intermitant scheme
# with hidden units, only readout from some-> enforce regions?
# annealing or correcting for importance between MSE and CE losses
# low-rank and feedback? --> analyze once it seems to work!; or maybe fix specific target pattern?
###

# %% analyze bistable results

# %% study noise effects!
