# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 23:24:23 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

# %% SRM
def RaisedCosine_basis(nkbins,nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+6) # spacing between bumps
    width = 3.*dbcenter # width of each bump
    bcenters = 1.1*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm).T

#RaisedCosine_basis(20,5)
    
def delay_function(n,k,tl):
    """
    More biophysical delayed function, given a width parameter n, location of kernel k,
    and the time window tl (n=5-10 is a normal choice)
    """
    beta = np.exp(n)
    fkt = beta*(tl/k)**n*np.exp(-n*(tl/k))
    return fkt

plt.figure()
for ii in range(0,10):
    plt.plot(delay_function(5,ii,np.arange(0,10,0.1)))

# %%
def neuralGLM_net(k,n,h,S):
    """
    neural GLM with k stimulus, n post-spike history with coupling filter, and h-step kernels
    for N neurons and parameterized by d basis functions
    system is driven by stimulus time series
    k: N x d
    n: N x N x d
    h: 1  (length of the kernel)
    S: N x T
    """
    N, T = S.shape[0], S.shape[1]  #space and time
    us = np.zeros((N,T+h))  #all activity through time
    d = k.shape[1]  #the weights on basis for non-Markovian window
    basis = RaisedCosine_basis(h,d)  #(h x d) share an basis function (can change here)
    
    for tt in range(h,T):
        ut = np.einsum('ij,ij->i', S[:,tt-h:tt], np.dot(k,basis.T) ) + \
             np.einsum('ijk,jk->i',  np.dot(n,basis.T), us[:,tt-h:tt])
        #ut[ut<0] = 0
        #ut[ut>100] = 100
        ut = 100/(1+np.exp(ut))
        us[:,tt] = np.random.poisson(ut)
    return us

def control_net(k,n,S):
    """
    A control for the instantaneous connected neural network with Poisson firing, with 
    stimulus projection k, adjecency matrix n, and stimulus S
    k: N
    n: N x N
    S: N x T
    """
    N, T = S.shape[0], S.shape[1]  #space and time
    us = np.zeros((N,T))  #all neural activities through time
    for tt in range(1,T-1):
        ut = us[:,tt-1] + S[:,tt-1]*k + np.matmul(n, us[:,tt-1])
        #ut[ut<0] = 0
        #ut[ut>100] = 100
        ut = 100/(1+np.exp(ut))
        us[:,tt] = np.random.poisson(ut)
    return us

# %% simulation for all random parameters
N = 50   #neurons
T = 3000 #time
d = 5    #basis functions
h = 20   #length of history

###response kernel
K_ = np.random.randn(N,d)*1.
#K_ = np.repeat(-np.array([-1,-0.75,-0.5,-0.25,-0.1])[None,:],N,axis=0)
#K_ = np.repeat(np.random.randn(5)[None,:],N,axis=0)

###connectivity kernels
N_ = np.random.rand(N,N,d)*.1
N_[np.arange(0,N),np.arange(0,N),:] = -np.random.rand(N,d)*.2
#J = np.random.randint(-1,2,(N,N))
#J[J==0] = -1
J = np.ones((N,N))
N_ = np.repeat(np.expand_dims(J,axis=2),d,axis=2)*N_

stim = np.random.randn(N,T)*.1
stim[:,1000:1500] = np.random.randn(N,500)*1.
#stim[:,2000:2500] = 0.

us = neuralGLM_net(K_,N_,h,stim)
plt.subplot(221)
plt.imshow(us,aspect='auto')
plt.subplot(222)
plt.plot(us.T)

us2 = control_net(K_[:,0], N_.mean(2)*10.,stim)
plt.subplot(223)
plt.imshow(us2, aspect='auto')
plt.subplot(224)
plt.plot(us2.T)

# %% building attractor structures
# %%
### symetric
N_ = np.random.randn(N,N,d)*1
updipos = np.triu_indices(N,k=0,m=N)
temp = N_.copy()
temp[updipos[0],updipos[1],:] = temp[updipos[1],updipos[0],:]
N_ = temp.copy()
us = neuralGLM_net(K_,N_,h,stim)
plt.imshow(us,aspect='auto')

# %%
### E-I
N_ = np.random.randn(N,N,d)*.01
nIs = np.random.randint(0,N,int(N*0.25))  #select some inhibitory neurons
N_[nIs,:,:] = -np.abs(N_[nIs,:,:])*1
us = neuralGLM_net(K_,N_,h,stim)
plt.imshow(us,aspect='auto')

# %% state-dependency
