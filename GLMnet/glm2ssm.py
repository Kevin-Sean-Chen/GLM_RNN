# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 00:22:43 2023

@author: kevin
"""


###
# simulate population Poisson spikes that have stochastic response to deterministic input
# train the GLM-RNN to produce probablistic output
# then fit ssm to prove that it indeed forms state-dependent input-out properties
###

import matplotlib.pyplot as plt
import ssm
#from ssm.util import one_hot, find_permutation

import autograd.numpy as np
import numpy.random as npr

from glmrnn.glm_obs_class import GLM_PoissonObservations
from glmrnn.glmrnn import glmrnn
from glmrnn.target_spk import target_spk

from sklearn.cluster import KMeans
import random

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% setup network parameters
N = 5
T = 200
dt = 0.1
tau = 2
### setup network
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
### setup target spike pattern
d = 1  # latent dimension
my_target = target_spk(N, T, d, my_glmrnn)

#my_target.M = np.ones((N,1))*3
my_target.M *= 2#.5

# %% produce target spikes
prob = 0.5
targ_spk, targ_latent = my_target.stochastic_rate(prob)

plt.figure()
plt.imshow(targ_spk, aspect='auto')

# %% test with random input
#input sequence
num_sess = 100 # number of example sessions

inpts_ = np.zeros(T)[:,None]
inpts_[int(T/2):] = prob
inpts_50 = np.zeros(T)[:,None]
inpts_50[int(T/2):] = 0.5

inpts = np.repeat(inpts_[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

# %% generate training sets
true_latents, true_spikes, true_ipt = [], [], []
for sess in range(num_sess):
    true_y, true_z = my_target.stochastic_rate(prob)
    true_spikes.append(true_y.T)
    true_latents.append(true_z)
    true_ipt.append(inpts[sess])
    
    ### for mixed learning
    true_y, true_z = my_target.stochastic_rate(0.75)
    true_spikes.append(true_y.T)
    true_latents.append(true_z)
    true_ipt.append(inpts_50*2 *0.75)
    true_y, true_z = my_target.stochastic_rate(0.25)
    true_spikes.append(true_y.T)
    true_latents.append(true_z)
    true_ipt.append(inpts_50*2 *0.25)  
    ###
    
# %% inference
datas = (true_spikes, true_ipt)
my_glmrnn.T = 200
my_glmrnn.lamb = 2
my_glmrnn.fit_glm(datas)  # using ssm gradient

# %%
ii = 0
spk,rt = my_glmrnn.forward(true_ipt[ii]*1)
#my_glmrnn.noise = my_glmrnn.b*2. #np.mean(true_spikes[0],0)*9 #
#my_glmrnn.W *= 5
#spk,rt = my_glmrnn.forward_rate(true_ipt[ii])

plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(true_spikes[ii].T,aspect='auto')
plt.title('true spikes',fontsize=40)
plt.subplot(122)
plt.imshow(spk,aspect='auto')
plt.title('inferred spikes',fontsize=40)

# %% spiking pattern analysis
def pattern_m(r1,r2):
    v1,v2 = r1.reshape(-1), r2.reshape(-1)
    m = np.corrcoef(v1,v2)[0][1]
    return m
rep = 100
sim_spk = []
#sim_rt = []
pattern_spk = []
m_pattern = []  # overlap for two patterns across sessions
for rr in range(rep):
    spk,rt = my_glmrnn.forward(true_ipt[150])  # fixed or vary across trials
    sim_spk.append(spk)
    spk50,rt50 = my_glmrnn.forward(inpts_50)  # for comparison
    pattern_spk.append(spk50)
    template = true_spikes[rr].T
    m_pattern.append(pattern_m(spk[:,T//2:], template[:,T//2:])) ### fix this...should sort then compute
    
X_rand = np.array(pattern_spk).reshape(rep,-1)
X_test = np.array(sim_spk).reshape(rep,-1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_rand)  # fit cluster to 50/50 case
predvec = kmeans.predict(X_test)  # test with biased generative model
print(np.sum(predvec)/rep)

# %%
plt.hist(np.array(m_pattern))
plt.xlabel('pattern correlation',fontsize=30)
plt.ylabel('count', fontsize=30)

# %% long-term simulation with trained network
# hypothesis: switching states responding to the same input!!
rep_stim = 20
long_ipt = np.tile(true_ipt[0]*1,rep_stim).T.reshape(-1)[:,None]
#long_ipt = np.array(random.sample(true_ipt, rep_stim)).reshape(-1)[:,None]
my_glmrnn.T = len(long_ipt)
spk, rt = my_glmrnn.forward(long_ipt)
plt.figure(figsize=(15,10))
plt.subplot(4,1,(1,3))
plt.imshow(spk, aspect='auto')
plt.subplot(4,1,4)
plt.plot(long_ipt)
plt.xlim([0,len(long_ipt)])

# %% plot simple psychometric curve
prob_vec = np.array([0.1, 0.3, 0.5,0.7,0.9])
res_prob = np.array([[0.02,0,0],[0.24,0.26,0.11],[0.45,0.5,0.62],[0.78,0.76,0.68],[1,0.96,0.99]])
plt.plot(prob_vec, res_prob,'ko')
plt.xlabel('input strength',fontsize=30)
plt.ylabel('choice probability', fontsize=30)

# %% generate spikes from trained network and fit with GLM-HMM
# %% ##########################################################################
# %% collect data
true_spikes_ssm = []
inpts_ssm = []
rep = 20
rep_stim = 20
for rr in range(rep):
    long_ipt = np.array(random.sample(true_ipt, rep_stim)).reshape(-1)[:,None]
    my_glmrnn.T = len(long_ipt)
    spk, rt = my_glmrnn.forward(long_ipt)
    true_spikes_ssm.append(spk.T)
    inpts_ssm.append(long_ipt)
    
# %% inference
num_states = 2
obs_dim = N*1
input_dim = 1
inf_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations= "poisson", transitions="inputdriven")
inf_glmhmm.observations = GLM_PoissonObservations(num_states, obs_dim, input_dim) ##obs:"input_driven"

N_iters = 100 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = inf_glmhmm.fit(true_spikes_ssm, inputs=inpts_ssm, method="em", num_iters=N_iters)

# %% analyze posterior fits
posterior_probs = [inf_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(true_spikes_ssm, inpts_ssm)]

# %%
sess_id = 1 #session id; can choose any index between 0 and num_sess-1
plt.figure(figsize=(15,10))
for k in range(num_states):
    plt.plot(posterior_probs[sess_id][:, k], label="State " + str(k + 1), lw=2)
plt.ylim((-0.01, 1.01))
plt.yticks([0, 0.5, 1])
plt.xlabel("time", fontsize = 30)
plt.ylabel("p(state)", fontsize = 30)

plt.figure(figsize=(15,10))
plt.subplot(4,1,(1,3))
plt.imshow(true_spikes_ssm[sess_id].T, aspect='auto')
plt.subplot(4,1,4)
plt.plot(inpts_ssm[sess_id])
plt.xlim([0,len(long_ipt)])

plt.figure()
plt.plot(inf_glmhmm.observations.Wk[:,:,0].T)
plt.xlabel('neuron', fontsize=30)
plt.ylabel('weights', fontsize=30)
plt.legend(['state 1', 'state 2'], fontsize=20)