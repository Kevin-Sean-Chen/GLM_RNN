# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 02:22:21 2020

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

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

# %% basis function
def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*0.5+0.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] #constructing bases
    return np.array(BBstm).T  #time x basis

plt.plot(basis_function1(150,6))

def flipkernel(k):
    """
    flipping kernel to resolve temporal direction
    """
    return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
def kernel(theta, pad):
    """
    Given theta weights and the time window for padding,
    return the kernel contructed with basis function
    """
    nb = len(theta)
    basis = basis_function1(pad, nb)  #construct basises
    k = np.dot(theta, basis.T)  #construct kernels with parameter-weighted sum
    return flipkernel(k)

# %% Poisson GLM
def NL(x):
    """
    Nonlinearity for Poisson GLM
    """
    return 1/(1+np.exp(-x))  #np.exp(x)  #

def Pois_spk(lamb, delt):
    """
    Poisson process for spiking
    """
    y= np.random.poisson(lamb*delt)
    return y

def coupled_GLM():
    spks = 0
    return spks

# %%
#network settings
def GLM_cir(ss,gg):
    nn = 3  #number of neurons in the circuit
    pad = 100  #time window used for history-dependence
    T = 10000  #time steps
    deltt = 0.01  #time bins
    spks = np.zeros((nn,T))  #recording all spikes
    Psks = np.zeros((nn,T))  #recording all spiking probability
    #GLM settings
    nb = 6  #number of basis function used to construct kernel
    ks = np.random.rand(nn,nb)  #stimulus filter for each neuron
    hs = np.random.randn(nn,nn,nb)  #coupling filter between each neuron and itself
    v1 = np.array([1,0,1])#np.random.randn(nneuron)
    v2 = np.array([0,0,1])
    v3 = np.array([1,1,0])
    ww = gg*(np.outer(v1,v1) + np.outer(v2,v2) + np.outer(v3,v3) + np.random.randn(nn,nn)*.1)
    np.fill_diagonal(ww,-ss*np.ones(nn))
    hs[:,:,1] = ww
    Ks = np.array([kernel(ks[kk,:],pad) for kk in range(0,nn)])
    Hs = np.array([kernel(hs[ii,jj,:],pad) for ii in range(0,nn) for jj in range(0,nn)]).reshape(nn,nn,pad)
    mus = np.random.randn(nn)*0.1  #fiting backgroun
    #stimulus (noise for now)
    It = np.random.randn(nn,T)*.5 + .5*np.repeat(np.sin(np.linspace(0,T,T)/200),nn).reshape(T,nn).T
    It = It*.05
    for tt in range(pad,T):
        Psks[:,tt] = NL(np.sum(Ks*It[:,tt-pad:tt],axis=1) + mus + \
            np.einsum('ijk,ik->i',  Hs, spks[:,tt-pad:tt]) )
        spks[:,tt] = Pois_spk(Psks[:,tt], deltt)
    return Psks, spks

# %%
Psks, spks = GLM_cir(10,1)
plt.figure()
plt.subplot(211)
plt.imshow(Psks, aspect='auto')
plt.subplot(212)
plt.imshow(spks, aspect='auto')

# %%
plt.figure()
Psks, spks = GLM_cir(-1,10)
plt.subplot(221)
plt.imshow(Psks, aspect='auto')
plt.subplot(222)
Psks, spks = GLM_cir(-1,-10)
plt.imshow(Psks, aspect='auto')
plt.subplot(223)
Psks, spks = GLM_cir(-10,10)
plt.imshow(Psks, aspect='auto')
plt.subplot(224)
Psks, spks = GLM_cir(-10,-10)
plt.imshow(Psks, aspect='auto')

# %%
plt.figure()
Psks, spks = GLM_cir(1,10)
plt.subplot(221)
plt.imshow(spks, aspect='auto')
plt.subplot(222)
Psks, spks = GLM_cir(1,-10)
plt.imshow(spks, aspect='auto')
plt.subplot(223)
Psks, spks = GLM_cir(10,10)
plt.imshow(spks, aspect='auto')
plt.subplot(224)
Psks, spks = GLM_cir(10,-10)
plt.imshow(spks, aspect='auto')
