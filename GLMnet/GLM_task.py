# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:44:07 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

from pyglmnet import GLM, simulate_glm
from pyglmnet import GLMCV
from pyglmnet import GLM

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

# %%
###############################################################################
# %% N-bit task
dt = 0.1  #ms
T = 10000  #total time
time = np.arange(0,T,dt)   #time axis
lt = len(time)
pulse_w = 10
pulse_A = 1
n_pulse = 6
noise = 0.01
### target I/O signal
time_points = np.random.choice(lt,n_pulse)  #4 points, 2 for ON 2 for OFF
input_ = np.zeros(lt)
for p in time_points:
    input_[int(p):int(p)+pulse_w] = pulse_A*np.random.choice(np.array([1,-1]))
output_ = np.zeros_like(input_)
output_[0] = np.random.choice(np.array([1,-1]))  #random initial condition
for t in range(lt-1):
    if input_[t] == 0:
        output_[t+1] = output_[t]
    elif input_[t]>0:
        output_[t+1] = 1
    elif input_[t+1]<0:
        output_[t+1] = -1
output_ = output_ + np.random.randn(lt)*noise

plt.plot(input_)
plt.plot(output_)

# %% GLM net fitting
#fake spike train output
rate = np.concatenate((input_,-input_,output_)).reshape(3,lt)  ### input signal, output, and negative output
nneuron = 0
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
Y = np.squeeze(rate[nneuron,:])  #spike train of interest
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
stimulus = input_[:,None]  #same stimulus for all neurons
X = build_convolved_matrix(stimulus, rate.T, Ks, couple)  #design matrix with features projected onto basis functions
###pyGLMnet function with optimal parameters
distr = "binomial"
glm = GLMCV(distr=distr, tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)

# %% direct simulation
yhat = simulate_glm(distr, glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
plt.figure()
plt.plot(Y*1.,label='input')  #ground truth
plt.plot(yhat,'--',label='ouput_est')
plt.plot(-output_,'k',alpha=0.5,label='target')
plt.legend()

# %%
theta = glm.beta_
dc_ = theta[0]
theta_ = theta[1:]
if couple == 1:
    theta_ = theta_.reshape(nbasis,N+1)  #nbasis times (stimulus + N neurons)
    allKs = np.array([theta_[:,kk] @ Ks for kk in range(N+1)])
elif couple == 0:
    allKs = Ks.T @ theta_

plt.figure()
plt.plot(allKs.T)
