# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 21:23:22 2023

@author: kevin
"""
###
# demo srcipt to simulate GLM-HMM with Poisson observations, 
# fit a GLM-RNN to produce this spiking pattern,
# then do inference to recover GLM-HMM parameters.
# if this works, we can now study how the stochastic network implments computation.
###

import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation

import autograd.numpy as np
import numpy.random as npr

from glmrnn.glm_obs_class import GLM_PoissonObservations
from glmrnn.glmrnn import glmrnn

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)


# %%
num_states = 2        # number of discrete states           K
obs_dim = 10          # number of observed dimensions       D
input_dim = 1         # input dimensions                    M

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="poisson", transitions="inputdriven")  
# observation class would be replaced to GLM_Pousson
# transitions: sticky , standard , inputdriven
# possible to be both driven!

# input driving conditions
driven_emission = 1
driven_state = 1
drive_logic = np.min([driven_emission+driven_state,1])

# %%  replace from here for now
if driven_emission==1:
    true_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim)
    print(true_glmhmm.observations.Wk.shape)
if driven_state==1 :
    true_glmhmm.transitions.Ws *= 3
    print(true_glmhmm.transitions.Ws)
else:
    print(true_glmhmm.transitions.transition_matrix)

# %%
#input sequence
num_sess = 10 # number of example sessions
time_len = 1000 # time duration of simulation
inpts = np.ones((num_sess, time_len, input_dim)) # initialize inpts array
stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]

### discrete input
#inpts[:,:,0] = np.random.choice(stim_vals, (num_sess, time_len)) # generate random sequence of stimuli
#inpts[:,:,1] = np.random.choice(stim_vals, (num_sess, time_len)) 
### random input
#inpts = np.random.randn(num_sess, num_trials_per_sess, input_dim)
### noisy sine input
inpts = np.sin(2*np.pi*np.arange(time_len)/300)[:,None]*.5 +\
        np.sin(2*np.pi*np.arange(time_len)/1000)[:,None]*1. +\
        .1*npr.randn(time_len,input_dim)

inpts = np.repeat(inpts[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

# %%
# Generate a sequence of latents and choices for each session
true_latents, true_spikes = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(time_len, input=inpts[sess]*drive_logic)  #changed hmm.py line206!
    true_latents.append(true_z)
    true_spikes.append(true_y)
    
# %%
# Calculate true loglikelihood
true_ll = true_glmhmm.log_probability(true_spikes, inputs=inpts*drive_logic) 
print("true ll = " + str(true_ll))

# %% fix this~~~ for inference
############################################################################### "input_driven_obs"
new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations= "poisson", transitions="inputdriven")
if driven_emission==1:
    new_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim) ##obs:"input_driven"

N_iters = 100 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(true_spikes, inputs=inpts, method="em", num_iters=N_iters)#, tolerance=10**-4)

# %%
new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_spikes[0], input=inpts[0]*drive_logic)))

# %%
try: 
    true_obs_ws = true_glmhmm.observations.Wk #mus
    inferred_obs_ws = new_glmhmm.observations.Wk
    
    cols = ['r', 'g', 'b']
    plt.figure()
    for ii in range(num_states):
        plt.plot(true_obs_ws[:][ii],linewidth=5, label='ture', color=cols[ii])
        plt.plot(inferred_obs_ws[:][ii],'--',linewidth=5,label='inferred',color=cols[ii])
    plt.legend(fontsize=20)
    plt.title('Emission weights', fontsize=40)
except:
    print('no emission kernel')

# %%
try:
    true_glmhmm.transitions.Ws
    true_trans_ws = true_glmhmm.transitions.Ws
    inferred_trans_ws = new_glmhmm.transitions.Ws
    
    plt.figure()
    for ii in range(num_states):
        plt.plot(true_trans_ws[ii],'*',markersize=15, label='ture', color=cols[ii])
        plt.plot(inferred_trans_ws[ii],'o',markersize=15,label='inferred', color=cols[ii])
    #plt.legend(fontsize=40)
    plt.title('Transition weights', fontsize=40)
except:
    print('no transition kernel')

# %% introducing GLM-RNN
###############################################################################
# %% GLM-RNN model
N = obs_dim*1
T = time_len*1
dt = 0.1
tau = 2

spk_targ = true_spikes[0].T
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
spk,rt = my_glmrnn.forward(inpts[0])

# %% inference
data = (spk_targ, inpts[0])
my_glmrnn.fit_single(data,lamb=0)

# %%
ii = 2
spk,rt = my_glmrnn.forward(inpts[ii])
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(true_spikes[ii].T,aspect='auto')
plt.title('true spikes',fontsize=40)
plt.subplot(122)
plt.imshow(spk,aspect='auto')
plt.title('inferred spikes',fontsize=40)

# %% test with batch
#datas = ([true_spikes[0]], [inpts[0]])  # debug this~~~   # might be 'dt'??
datas = (true_spikes, inpts)
#my_glmrnn.fit_batch(datas)  # using regression tools
#my_glmrnn.fit_batch_sp(datas)  # this seems to currently work!!...but take too long
my_glmrnn.fit_glm(datas)  # using ssm gradient

# %% now fit it back with ssm!
###############################################################################
rnn_spikes = []
for sess in range(num_sess):
    spk, rt = my_glmrnn.forward(inpts[sess])  #changed hmm.py line206!
    rnn_spikes.append(spk.T)
    
# %%
rnn_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations= "poisson", transitions="inputdriven")
if driven_emission==1:
    rnn_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim)
fit_ll = rnn_glmhmm.fit(rnn_spikes, inputs=inpts, method="em", num_iters=N_iters)
#fit_ll = rnn_glmhmm.fit(true_spikes, inputs=inpts, method="em", num_iters=N_iters) #positive control...

# %% emissions
rnn_glmhmm.permute(find_permutation(true_latents[0], rnn_glmhmm.most_likely_states(rnn_spikes[0], input=inpts[0])))
true_obs_ws = true_glmhmm.observations.Wk #mus
inferred_obs_ws = rnn_glmhmm.observations.Wk

cols = ['r', 'g', 'b']
plt.figure()
for ii in range(num_states):
    plt.plot(true_obs_ws[:][ii],linewidth=5, label='ture', color=cols[ii])
    plt.plot(inferred_obs_ws[:][ii],'--',linewidth=5,label='inferred',color=cols[ii])
plt.legend(fontsize=20)
plt.title('Emission weights', fontsize=40)

# %% latents
ii = 0
inferred_states = rnn_glmhmm.most_likely_states(rnn_spikes[ii], input=inpts[ii])
plt.figure()
plt.subplot(211)
plt.imshow(true_latents[ii][None,:], aspect='auto')
plt.title('true latent',fontsize=40)
plt.subplot(212)
plt.title('inferred latent',fontsize=40)
plt.imshow(inferred_states[None,:], aspect="auto")
