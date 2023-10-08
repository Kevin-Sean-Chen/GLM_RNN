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
from glmrnn import glmrnn
from target_spk import target_spk

from sklearn.cluster import KMeans

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
#my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='basis', nl_type='sigmoid', spk_type="Poisson")
### setup target spike pattern
d = 1  # latent dimension
my_target = target_spk(N, T, d, my_glmrnn)

#my_target.M = np.ones((N,1))*3
my_target.M *= 2#.5
my_glmrnn.lamb_max = 10
# %% produce target spikes
### bistable, oscillation, chaotic, sequence, line_attractor, brunel_spk
#targ_spk, targ_latent = my_target.sequence(50)
#targ_spk, targ_latent = my_target.chaotic()
#targ_spk, targ_latent = my_target.bistable()
#targ_spk, targ_latent = my_target.oscillation(10)
#targ_spk, targ_latent = my_target.line_attractor(5)
#targ_spk, targ_latent = my_target.brunel_spk('SR', 10)

prob = 0.5
targ_spk, targ_latent = my_target.stochastic_rate(prob)

plt.figure()
plt.imshow(targ_spk, aspect='auto')

# %% test with random input
#input sequence
num_sess = 100 # number of example sessions
input_dim = 1
inpts_ = np.sin(2*np.pi*np.arange(T)/600)[:,None]*.5 +\
        np.cos(2*np.pi*np.arange(T)/300)[:,None]*1. +\
        .1*npr.randn(T,input_dim)\
        + np.linspace(-2,2,T)[:,None]

inpts_ = np.zeros(T)[:,None]
inpts_[int(T/2):] = prob
inpts_50 = np.zeros(T)[:,None]
inpts_50[int(T/2):] = 0.5

inpts = np.repeat(inpts_[None,:,:], num_sess, axis=0)
inpts = list(inpts) #convert inpts to correct format

###
# test with 'clock' signal for autonomous system
###
# %% generate training sets
true_latents, true_spikes, true_ipt = [], [], []
for sess in range(num_sess):
#    true_y, true_z = my_target.bistable() #
#    true_y, true_z = my_target.sequence(50)  # maybe fix this to pass latent type as string~
#    true_y, true_z = my_target.oscillation(10)
#    true_y, true_z = my_target.line_attractor(5)
#    true_y, true_z = my_target.brunel_spk('SR', 10)  #SR, AI, SIf, SIs
    true_y, true_z = my_target.stochastic_rate(prob)
    
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
my_glmrnn.lamb = 0
my_glmrnn.fit_single(data,lamb=0)

# %%
ii = 5
#spk,rt = my_glmrnn.forward(true_ipt[ii])
my_glmrnn.noise = my_glmrnn.b*2. #np.mean(true_spikes[0],0)*9 #
#my_glmrnn.W *= 5
spk,rt = my_glmrnn.forward_rate(true_ipt[ii])

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
my_glmrnn.T = 200
my_glmrnn.lamb = 2
my_glmrnn.fit_glm(datas)  # using ssm gradient

# %% test states
#datas = (true_spikes, true_ipt, true_latents)
#my_glmrnn.fit_glm_states(datas,2)
###

# %% spiking pattern analysis
def pattern_m(r1,r2):
#    temp = r1*r2
    v1,v2 = r1.reshape(-1), r2.reshape(-1)
    m = np.corrcoef(v1,v2)[0][1]
    return m
rep = 100
sim_spk = []
#sim_rt = []
pattern_spk = []
m_pattern = []  # overlap for two patterns across sessions
for rr in range(rep):
    spk,rt = my_glmrnn.forward(true_ipt[0])  # fixed or vary across trials
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
my_glmrnn.T = len(long_ipt)
spk, rt = my_glmrnn.forward(long_ipt)
plt.figure(figsize=(15,10))
plt.subplot(4,1,(1,3))
plt.imshow(spk, aspect='auto')
plt.subplot(4,1,4)
plt.plot(long_ipt)
plt.xlim([0,len(long_ipt)])

###
# current important factors: training probablity, regulairzation, and maybe latents!?
# analysis with tuning curves and state-space fitting
###

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% use rate RNN and Poisson log-likelihood
from rnn_torch import RNN, lowrank_RNN, observed_RNN, RNNTrainer
import torch

# %% setup target
N = 10
T = 200
dt = 0.1
tau = 1
my_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson")
d = 1  # latent dimension
my_target = target_spk(N, T, d, my_glmrnn)
#my_target.M = np.ones((N,1))

num_sess = 100
true_latents, true_spikes, true_ipt = [], [], []
#inpts_ = np.arange(T,0,-1)[:,None]#np.sin(2*np.pi*np.arange(T)/100)[:,None]*.1
inpts = np.repeat(inpts_[None,:-1,:], num_sess, axis=0)
inpts = list(inpts)
for sess in range(num_sess):
#    true_y, true_z = my_target.sequence(20)  # maybe fix this to pass latent type as string~
#    true_y, true_z = my_target.chaotic()
#    true_y, true_z = my_target.oscillation(7)
#    true_y, true_z = my_target.bistable()  # does NOT work!
#    true_y, true_z = my_target.line_attractor(5)
    true_y, true_z = my_target.stochastic_rate(prob)
    
    true_r = my_glmrnn.kernel_filt(true_y)
    true_spikes.append(true_y.T)  # kernel filtered rate pattern
    true_latents.append(true_r/np.max(true_r))  # normalization
    true_ipt.append(inpts[sess])

# %% tensorize
target_spikes = torch.Tensor(np.array(true_spikes))
target_rates = torch.Tensor(np.transpose(np.array(true_latents),axes=(0, 2, 1)))
inp = torch.Tensor(np.array(true_ipt))

# %% training
inf_net = observed_RNN(1, N, dt, 1) 
masks = torch.ones(num_sess, T+0, N)
trainer = RNNTrainer(inf_net, 'MSE', spk_target=target_spikes)
losses = trainer.train(inp, target_rates, masks, n_epochs=150, lr=1e-1, batch_size=5)
### still need to fix poisson ll!

plt.plot(np.arange(len(losses)), losses)
plt.title('Learning curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %% generative with rate
trid = 10
_, outputs_rt = inf_net.forward(inp)
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.title('true activity',fontsize=40)
plt.imshow(outputs_rt[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')
plt.subplot(122)
plt.title('inferred activity',fontsize=40)
plt.imshow(target_rates[trid,:,:].T.detach().numpy().squeeze(), aspect='auto')

# %% generative with spikes
lamb = 5
gen_glmrnn = glmrnn(N, T, dt, tau, kernel_type='tau', nl_type='sigmoid', spk_type="Poisson")
gen_glmrnn.W = inf_net.J.detach().numpy()*lamb
gen_glmrnn.U = inf_net.B.detach().numpy().squeeze()*lamb
gen_glmrnn.b = gen_glmrnn.b*lamb
gen_glmrnn.lamb_max = 20
ii = 1
spk,rt = gen_glmrnn.forward(inp[ii].detach().numpy())
plt.figure(figsize=(15,10))
plt.subplot(121)
plt.imshow(target_rates[ii].T,aspect='auto')
plt.title('true spikes',fontsize=40)
plt.subplot(122)
plt.imshow(rt,aspect='auto')
plt.title('inferred spikes',fontsize=40)