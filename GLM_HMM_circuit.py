#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:44:01 2020

@author: kschen
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dotmap import DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% bastable circuit
###settings
dt = 0.1  #ms
T = 1000
time = np.arange(0,T,dt)
lt = len(time)

x = np.zeros((4,lt))  #voltage
spk = np.zeros_like(x)  #spikes
syn = np.zeros_like(x)  #synaptic efficacy
x[:,0] = np.random.randn(4)
J = np.array([[0, 1, -1, 0],\
              [0, 1.2,-1.3, 1],\
              [0,-1.1, 1.2, 1],\
              [0, 0, 0, 0]])
J = J.T*.1
#J = np.random.rand(4,4)
noise = 0.5
stim = np.random.randn(lt)*.5
taum = 5
taus = 10
E = 1.

eps = 10**-15
def LN(x):
    """
    Logistic nonlinearity
    """
    return np.random.poisson(100/(1+np.exp(-x*1.+eps)))

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt*( -x[:,tt]/taum + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[tt]*np.array([1,0,0,0]) + noise*np.random.randn(4)*np.sqrt(dt))
    spk[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = syn[:,tt] + dt*( (1-syn[:,tt])/taus - spk[:,tt]*E )
    
plt.figure() 
plt.subplot(211)
plt.imshow(spk,aspect='auto');
plt.subplot(212)
plt.plot(time,x.T);

# %% let's fit GLM-net!!!
def neglog(theta,Y,X):
    k = kernel(theta)
    nl = np.matmul(Y.T, np.log(LN(np.matmul(X,k)))) - np.sum(LN(np.matmul(X,k)))
    return nl
def kernel(theta):
    basis = basis_function()
    return k

def basis_function():
    return basis

def build_matrix(stimulus, spikes):
    n,t = spikes.shape()
    
    return Y, X

Y, X = build_matrix(stim,spk)
#pars = 
theta0 = np.random.randn()
res = sp.optimize.minimize( neglog, theta0, args=())#
