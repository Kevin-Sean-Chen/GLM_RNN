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
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="gaussian", transitions="standard")  
# observation class would be replaced to GLM_Pousson
# transitions: sticky , standard , inputdriven
# possible to be both driven!

# %%  replace from here for now
true_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim)
print(true_glmhmm.observations.Wk.shape)

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
inpts = np.sin(2*np.pi*np.arange(time_len)/100)[:,None]*.5+.1*npr.randn(time_len,input_dim)

inpts = np.repeat(inpts[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

# %%
# Generate a sequence of latents and choices for each session
true_latents, true_spikes = [], []
for sess in range(num_sess):
    true_z, true_y = true_glmhmm.sample(time_len, input=inpts[sess])  #changed hmm.py line206!
    true_latents.append(true_z)
    true_spikes.append(true_y)
    
# %%
# Calculate true loglikelihood
true_ll = true_glmhmm.log_probability(true_spikes, inputs=inpts) 
print("true ll = " + str(true_ll))

# %% fix this~~~ for inference
############################################################################### "input_driven_obs"
new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations= "gaussian", transitions="standard")
new_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim) ##obs:"input_driven"

N_iters = 100 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(true_spikes, inputs=inpts, method="em", num_iters=N_iters)#, tolerance=10**-4)

# %%
new_glmhmm.permute(find_permutation(true_latents[0], new_glmhmm.most_likely_states(true_spikes[0], input=inpts[0])))

# %%
true_obs_ws = true_glmhmm.observations.Wk #mus
inferred_obs_ws = new_glmhmm.observations.Wk

cols = ['r', 'g', 'b']
plt.figure()
for ii in range(num_states):
    plt.plot(true_obs_ws[:][ii],linewidth=5, label='ture', color=cols[ii])
    plt.plot(inferred_obs_ws[:][ii],'--',linewidth=5,label='inferred',color=cols[ii])
plt.legend(fontsize=20)
plt.title('Emission weights', fontsize=40)

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

# %%
###############################################################################
# %% GLM-RNN model
N = obs_dim*1
T = time_len*1
dt = 0.1
tau = 2

spk_targ = true_spikes[1].T
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
spk,rt = my_glmrnn.forward(inpts[1])

# %% inference
data = (spk_targ, inpts[1])
my_glmrnn.fit_single(data)

# %%
spk,rt = my_glmrnn.forward(inpts[1])
plt.figure()
plt.subplot(121)
plt.imshow(spk_targ,aspect='auto')
plt.subplot(122)
plt.imshow(spk,aspect='auto')

# %% test with batch
datas = (true_spikes, inpts)
my_glmrnn.fit_batch(datas)

# %%

