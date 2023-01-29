# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:00:27 2020

@author: kevin
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
import seaborn as sns
from ssm.util import one_hot, find_permutation

#%matplotlib inline

npr.seed(0)
sns.set(palette="colorblind")

# %%
## Set the parameters of the HMM
#time_bins = 1000 # number of time bins
#num_states = 2    # number of discrete states
#obs_dim = 1    # data dimension
#input_dim = 1    # input dimension
#num_categories = 3    # number of output types/categories
## Make an HMM
#true_hmm = ssm.HMM(num_states, obs_dim, input_dim, 
#               observations="categorical", observation_kwargs=dict(C=num_categories),
#               transitions="inputdriven")
#
## Optionally, turn up the input weights to exaggerate the effect
#true_hmm.transitions.Ws *= 3
## Create an exogenous input
#inpt = np.sin(2 * np.pi * np.arange(time_bins) / 50)[:, None] + 1e-1 * npr.randn(time_bins, input_dim)
## Sample some data from the HMM
#true_states, obs = true_hmm.sample(time_bins, input=inpt)
## Compute the true log probability of the data, summing out the discrete states
#true_lp = true_hmm.log_probability(obs, inputs=inpt)
## By default, SSM returns categorical observations as a list of lists.
## We convert to a 1D array for plotting.
#obs_flat = np.array([x[0] for x in obs])
## %%
## Plot the data
#plt.figure(figsize=(8, 5))
#plt.subplot(311)
#plt.plot(inpt)
#plt.xticks([])
#plt.xlim(0, time_bins)
#plt.ylabel("input")
#plt.subplot(312)
#plt.imshow(true_states[None, :], aspect="auto")
#plt.xticks([])
#plt.xlim(0, time_bins)
#plt.ylabel("discrete\nstate")
#plt.yticks([])
## Create Cmap for visualizing categorical observations
#plt.subplot(313)
#plt.imshow(obs_flat[None,:], aspect="auto", )
#plt.xlim(0, time_bins)
#plt.ylabel("observation")
#plt.grid(b=None)
#plt.show()
# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% load circuit results
obs = rr.T.copy() #rr[0,:][:,None] #
inpt = stim.T.copy()  #stim[0,:][:,None] #
# Set the parameters of the HMM
time_bins = obs.shape[1] # number of time bins
num_states = 3    # number of discrete states
obs_dim = obs.shape[1]    # data dimension
input_dim = stim.shape[0]    # input dimension
num_categories = 3    # number of output types/categories

# %% run driven HMM inference
# Now create a new HMM and fit it to the data with EM
N_iters = 100
hmm = ssm.HMM(num_states, obs_dim, input_dim, 
          observations="gaussian", #observation_kwargs=dict(C=num_categories),
          transitions="inputdriven")

# Fit
hmm_lps = hmm.fit(obs, inputs=inpt, method="em", num_iters=N_iters)

# %%
# Plot the log probabilities of the true and fit models
plt.figure()
plt.plot(hmm_lps, label="EM")
plt.legend(loc="lower right")
plt.xlabel("EM Iteration")
plt.xlim(0, N_iters)
plt.ylabel("Log Probability")
plt.show()

# %% plot results

inferred_state = hmm.most_likely_states(obs, input=inpt)

plt.figure()
plt.subplot(211)
plt.imshow(rr, aspect='auto');
plt.subplot(212)
plt.imshow(inferred_state[None,:],aspect='auto')


# %% condutional analysis
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
hmm_Ks = np.zeros((num_states, obs_dim+1, pad))
nneuron = 0
for ss in range(num_states):
    pos = np.where(inferred_state==ss)[0]
    rr_ = rr[:,pos]  #conditionally sample from one state
    Y = np.squeeze(rr_[nneuron,:])  #spike train of interest
    Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
    stimulus = stim[nneuron,pos][:,None]  #same stimulus for all neurons
    X = build_convolved_matrix(stimulus, rr_.T, Ks, couple)  #design matrix with features projected onto basis functions
    ###pyGLMnet function with optimal parameters
    glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
                score_metric="deviance",
                alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
    glm.fit(X, Y)
## %%   direct simulation
#    yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
#    plt.figure()
#    plt.plot(Y*1.)  #ground truth
#    plt.plot(yhat,'--')

## %%reconstruct kernel
    theta = glm.beta_
    dc_ = theta[0]
    theta_ = theta[1:]
    if couple == 1:
        theta_ = theta_.reshape(obs_dim+1,nbasis).T  #nbasis times (stimulus + N neurons)
        allKs = np.array([theta_[:,kk] @ Ks for kk in range(obs_dim+1)])
    elif couple == 0:
        allKs = Ks.T @ theta_

    hmm_Ks[ss,:,:] = allKs
    
#    plt.figure()
#    plt.plot(allKs.T)

# %% plotting
plt.figure()
for ss in range(num_states):
    plt.subplot(1,num_states,ss+1)
    plt.plot(hmm_Ks[ss,:,:].T)
    plt.title('state'+str(ss))

# %%
## Find a permutation of the states that best matches the true and inferred states
#hmm.permute(find_permutation(true_states, hmm.most_likely_states(obs, input=inpt)))
#inferred_states = hmm.most_likely_states(obs, input=inpt)
#
## %%
## Plot the true and inferred states
#plt.figure(figsize=(8, 3.5))
#plt.subplot(211)
#plt.imshow(true_states[None, :], aspect="auto")
#plt.xticks([])
#plt.xlim(0, time_bins)
#plt.ylabel("true\nstate")
#plt.yticks([])
#plt.subplot(212)
#plt.imshow(inferred_states[None, :], aspect="auto")
#plt.xlim(0, time_bins)
#plt.ylabel("inferred\nstate")
#plt.yticks([])
#plt.show()

# %%
# Plot the true and inferred input effects
plt.figure(figsize=(8, 4))

vlim = max(abs(true_hmm.transitions.log_Ps).max(),
           abs(true_hmm.transitions.Ws).max(),
           abs(hmm.transitions.log_Ps).max(),
           abs(hmm.transitions.Ws).max())

plt.subplot(141)
plt.imshow(true_hmm.transitions.log_Ps, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=1)
plt.xticks(np.arange(num_states))
plt.yticks(np.arange(num_states))
plt.title("True\nBaseline Weights")
plt.grid(b=None)


plt.subplot(142)
plt.imshow(true_hmm.transitions.Ws, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=num_states/input_dim)
plt.xticks(np.arange(input_dim))
plt.yticks(np.arange(num_states))
plt.title("True\nInput Weights")
plt.grid(b=None)


plt.subplot(143)
plt.imshow(hmm.transitions.log_Ps, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=1)
plt.xticks(np.arange(num_states))
plt.yticks(np.arange(num_states))
plt.title("Inferred\nBaseline Weights")
plt.grid(b=None)


plt.subplot(144)
plt.imshow(hmm.transitions.Ws, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=num_states/input_dim)
plt.xticks(np.arange(input_dim))
plt.yticks(np.arange(num_states))
plt.title("Inferred\nInput Weights")
plt.grid(b=None)
plt.colorbar()
plt.show()


