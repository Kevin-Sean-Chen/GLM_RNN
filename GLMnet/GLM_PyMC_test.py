# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:00:38 2021

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp  #.optimize.minimize, .special.gammaln, and .linalg.hankel
from scipy.linalg import hankel
from scipy.optimize import minimize
from scipy.special import gammaln

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 

import pymc3 as pm
from pymc3 import *
import arviz as az

# %% Simulate GLM spikes
# %% load parameters
# simulate Poisson GLM with autoregressive stimulus filter
w = np.array([ 2.5, 0.0, 0.0, 0.0, -0.01368046, -0.01986828,
 -0.03867417, -0.05218188, -0.10044614, -0.12434759, -0.19540891, -0.23327453,
 -0.32255702, -0.40001292, -0.46124429, -0.46235415, -0.43928836, -0.52066692,
 -0.58597496, -0.15804368,  1.2849799,   1.91338741,  1.10402054,  0.23188751,
  0.00331092, -0.0111924, ])

D = len(w)  #kernel length
T = 10000  #time bins
xx = np.random.randint(0,2,[T,])-0.5  #binary stimuli
xx = 0.3*np.random.randn(T)  #Gaussian stimuli
dtsp = 0.01  #spiking bin size

# %% Generate spikes
def PGLM_spk(xx,w,dtsp):
    D = len(w)
    T = len(xx)
    # generate design matrix 
    X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
    X = np.concatenate((np.ones([T,1]),X),axis=1)
    # generate spikes 
    y = np.random.poisson(np.exp(X @ w)*dtsp)
    return y, X

Y, X = PGLM_spk(xx,w,dtsp)
# look at data
plt.figure()
plt.subplot(2,1,1)
plt.plot(xx[:100])
plt.xlabel("time bin")
plt.ylabel("stimulus value")
plt.subplot(2,1,2)
plt.plot(Y[:100])
plt.xlabel("time bin")
plt.ylabel("spike count")

# %% PyMC inference
# %% create model
#D = Xb.shape[1]
sigma2 = 1.0
b = 1.0 / np.sqrt(2.0 * sigma2)
with pm.Model() as mdl_PGLM:

    # priors for weights
    #ww = pm.Normal("ww", mu=0, sigma=0.1, shape=D)
    ww = pm.Laplace('ww', mu=0, b=b, shape=D)

    # define linear model and exp link function
    theta = np.exp(Xb @ ww)

    ## Define Poisson likelihood
    y = pm.Poisson("y", mu=theta, observed=Y)
    
# %% sampling
with mdl_PGLM:
    inf_GLM = pm.sample(1000, tune=1000, cores=4, return_inferencedata=True)

# %%
idata = az.summary(inf_GLM)
kk = np.array(idata)
plt.figure()
plt.plot(kk[:,0])

# %%
postm = kk[:,0]
base = postm[0]
Ks = np.reshape(postm[1:],[int(len(postm[1:])/B),B])
Ks = Ks @ basis_set
plt.figure()
plt.plot(Ks.T)

