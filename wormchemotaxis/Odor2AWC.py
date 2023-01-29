#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:10:15 2019

@author: kschen
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math

import seaborn as sns
sns.set_style("white")
sns.set_context("talk")

# %%
def ODE_filter(stimuli, dt):
    """
    Input odor stimuli and output [Ca2+] signal after signal cascade in AWC neuron
    Kinetic parameters are from Kato's 2014 Neuron paper... additional nonlinearity and adaptation might be missing
    """
    ka = 1/2.99
    kf = 1/0.04
    ks = 1/10.88
    kas = 0.0024
    kaf = 1  #ralative value to kas...    
    ###time = 200  #sec
    ###dt = 0.01  #~10ms
    t = np.arange(0,len(stimuli))
    
    M = np.array([[-ka, 0, 0],[kaf, -kf, 0],[-kas, 0, -ks]])  #linear response matrix
    state = np.zeros((3,len(t)))  #all three states
    R = np.zeros(len(t))  #output
    
    for tt in range(0,len(t)-1):
        state[:,tt+1] = state[:,tt] + dt*(M @ state[:,tt]) + np.array([stimuli[tt],0,0])
        R[tt] = sum(state[1:,tt])
    return R

def GCaMP_LN(Ca2,dt):
    """
    Recive [Ca2+] and map to gCaMP activity
    Kernel and Hill function according to the original Tian 2009 nature paper
    """
    k = 1/0.4963  #decay exponent for GCaMP frin the original Tian 2009 Nature paper
    t = np.arange(0,len(Ca2))*dt
    Kgcamp = np.exp(-k*t)
    F = np.convolve(Ca2,Kgcamp,'same')
    ##### all the nonlinear parameters
    a = 1
    Fmin = -0.2
    p = 2.3  #Hill coefficient for GCaMP
    c = 0
    dFF = power_NL(F,a,Fmin,p,c)
    ##### Hill function way to go
    #Kd = 1  #dissociation constant(?)
    #dFF = F**n/(Kd**n+F**n)  #an actual Hill function (?)
    return dFF

def power_NL(F,a,Fmin,p,c):
    """
    Power-nonlinearity mentioned in the Kato paper
    """
    return a*(F-Fmin)**p + c

def inference_kernel(stim, R, win):
    """
    infer kerenl given stimulus and response time series
    """
    ## infer effective kernel
    X = np.zeros((len(stim)-win,win))  #design matrix
    #design matrix
    for xx in range(0,X.shape[0]):
        X[xx,:] = stim[xx:xx+win]
    #regression
    K_est = np.linalg.inv(X.T @ X) @ X.T @ R[win:]
    return K_est

# %% analytic solution for kernel
ka = 1/2.99
kf = 1/0.04
ks = 1/10.88
kas = 0.0024
kaf = 1  #ralative value to kas...
time = 10  #sec
dt = 0.01  #~10ms
tk = np.arange(0,time,dt)
Kt = kaf/(ka-kf)*np.exp(-kf*tk) - kas/(ka-ks)*np.exp(-ks*tk) + (kas/(ka-ks) - kaf/(ka-kf))*np.exp(-ka*tk)
plt.plot(tk,Kt)
plt.xlabel('time (s)')
plt.ylabel('Kernel')

# %% inference
## noise input
time = 500  #sec
dt = 0.01  #~10ms
t = np.arange(0,time,dt)## periodic input

f = 20  #frequency  (5Hz for 200ms flickering)
stim = np.cos(t*(2*np.pi)*f)*1 + 1*np.random.randn(len(t))

plt.plot(tk,Kt/np.linalg.norm(Kt),label='analytic')#/np.linalg.norm(Kt)
R = ODE_filter(stim, dt)
win = len(tk)
K_est = inference_kernel(stim,R,win)
tempK = np.flip(K_est)/np.linalg.norm(K_est)
#tempK = np.flip(K_est)
plt.plot(tk, tempK, '--',label='inferred')
plt.xlabel('time (s)')
plt.ylabel('AWC kernel')
plt.legend()
