# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 21:56:22 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 

# %% functions
### basis function, nonlinearity, and network
def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm).T

def basis_function2(n, k, tl):
    """
    More biophysical delayed function, given a width parameter n, location of kernel k,
    and the time window tl (n=5-10 is a normal choice)
    """
    beta = np.exp(n)
    tl = np.arange(tl)
    fkt = beta*(tl/k)**n*np.exp(-n*(tl/k))
    return fkt

def NL(x,spkM):
    """
    Passing x through logistic nonlinearity with spkM maximum
    """
    nl = spkM/(1+np.exp(x))
    #nl = np.tanh(x)
    return nl

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
#    N = len(x)
#    spike = np.random.rand(N) < x*dt*0.1
    x[x<0] = 0  #ReLU
    #x[x>100] = 100  #hard threshold in case of instability
    spike = np.random.poisson(x*dt)
    return spike

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

# %% dynamics
### GLM network simulation, with input
def GLM_net_stim(allK, dcs, S):
    """
    Simulate a GLM network given all response and coupling kernels and the stimulus
    """
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    _,T = S.shape  #all the same stimulus for now
    us = np.zeros((N,T))  #all activity through time
    spks = np.zeros((N,T))  #for spiking process
    K_stim = allK[:,0,:]  # N x pad response kernels
    K_couple = allK[:,1:,:]  # N x N xpad coupling filter between neurons (and itself)
    #K_couple.transpose(1, 0, 2).strides
    
    for tt in range(h,T):
        ut = np.einsum('ij,ij->i', S[:,tt-h:tt], (K_stim)) + \
             np.einsum('ijk,jk->i',  (K_couple), spks[:,tt-h:tt])  #neat way for linear dynamics
        ut = NL(ut + dcs, spkM)  #share the same nonlinearity for now
        us[:,tt] = ut #np.random.poisson(ut)
        spks[:,tt] = spiking(NL(ut + dcs, spkM), dt)  #Bernouli process for spiking
    return us, spks

def GLM_net(M_, allK, T):
    """
    Simulate a GLM network given all response and coupling kernels
    """
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    us = np.zeros((N,T))  #all activity through time
    spks = np.zeros((N,T))  #for spiking process
    kernels = np.einsum('ij,ijk->ijk', M_, allK)
    #K_couple.transpose(1, 0, 2).strides
    
    for tt in range(h,T):
        ut = np.einsum('ijk,jk->i',  kernels, spks[:,tt-h:tt])  #neat way for linear dynamics
        us[:,tt] = NL(ut, spkM) #np.random.poisson(ut)
        spks[:,tt] = spiking(NL(ut, spkM), dt)  #Bernouli process for spiking
    return us, spks

def G_FORCE_network(M_, spkM, allK, T):
    """
    Simulate a GLM network given all response and coupling kernels
    """
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    us = np.zeros((N,T))  #all activity through time
    spks = np.zeros((N,T))  #for spiking process
    
    for tt in range(h,T):
        r = np.einsum('ijk,jk->i',  allK, spks[:,tt-h:tt])  #neat way for linear dynamics
        us[:,tt] = NL(r, spkM) #np.random.poisson(ut)
        spks[:,tt] = spiking(M_ @ NL(r, spkM), dt)  #Bernouli process for spiking
    return us, spks

# %% ground truth GLM-net
N = 10
T = 1000
dt = 0.1
time = np.arange(0,T,dt)
stim = np.random.randn(len(time))*0.1
stimulus = 0*np.repeat(stim[:,None],10,axis=1).T #identical stimulus for all three neurons for now
nbasis = 5
spkM = 1
pad = 150
nkernels = N**2+N  #xN coupling and N stimulus filters
thetas = np.random.randn(nkernels, nbasis)  #weights on kernels
#Ks = basis_function1(pad, nbasis)
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((nkernels,pad))  #number of kernels x length of time window
for ii in range(nkernels):
    allK[ii,:] = np.dot(thetas[ii,:], Ks)
allK = allK.reshape(N,N+1,pad)
us, spks = GLM_net_stim(allK, 0, stimulus)

plt.figure()
plt.imshow(us,aspect='auto')

# %% Measurements
def autocorr_(x):
    result = np.correlate(x, x, mode='same')
    return result[int(result.size/2):]
def autocorr(x,lags):
    '''manualy compute, non partial'''
    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
#    corr=[np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    return np.array(corr)
def exp_fit(x,y):
    #A, tau = np.polyfit(x, np.log(y), 1)
    #np.polyfit(x, np.log(y), 1, w=np.sqrt(y))
    p0 = (1.,1.e-5,0.) # starting search koefs
    opt, pcov = curve_fit(model_func, x, y, p0)
    a, k, b = opt
    return k
def model_func(x, a, k, b):
    return a * np.exp(-k*x) + b

# %% GLM kernel setup
N = 100
nbasis = 5
pad = 100
spkM = 1
T = 2000
dt = 1
p_glm = 0.2
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((N,N,pad))  #number of kernels x length of time window
thetas = np.random.randn(N,N, nbasis)
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p_glm] = 0
mask[sparse<=p_glm] = 1
for ii in range(N):
    for jj in range(N):
        temp = np.dot(thetas[ii,jj,:], Ks)
        if ii==jj:
            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*10.
#            allK[ii,jj,:] = temp*mask[ii,jj]
        else:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*mask[ii,jj]
            
# %% parameter test
M_ = 1/np.sqrt(p_glm*N)*np.random.randn(N,N)
#us,spks = GLM_net(M_, allK, T)
us,spks = G_FORCE_network(M_, spkM, allK, T)
plt.figure()
plt.imshow(spks,aspect='auto')

# %% Dynamics
gs = np.array([0,0.001,0.01,0.1,1,10,100,1000])/1  #for recurrent strenght
gs = np.array([1,5,10,15,25,50,100])   #for max spike rate
win = 200
acs = np.zeros((len(gs),N,win))
Mtemp = np.random.randn(N,N)
for gg in range(len(gs)):
    M_ = Mtemp*1/np.sqrt(p_glm*N)  #1 or gs[gg]
    us,spks =G_FORCE_network(M_, gs[gg], allK, T)  #spkM or gs[gg]
    #GLM_net(M_, allK, T)
    for nn in range(N):
        acs[gg,nn,:] = autocorr(spks[nn,:], np.arange(win))

# %%
nn = len(gs)
color = iter(cm.rainbow(np.linspace(0, 1, nn)))
acss = np.zeros((len(gs),win))
for ii in range(len(gs)):
#    if ii==0:
#        acss[ii,:] = np.nanmean(acs[ii,:,:],axis=0)
#        plt.plot(acss[ii,:], 'k')
#    else:
#        acss[ii,:] = np.nanmean(acs[ii,:,:],axis=0)
#        c = next(color)
#        plt.plot(acss[ii,:], c=c)
    acss[ii,:] = np.nanmean(acs[ii,:,:],axis=0)
    c = next(color)
    plt.plot(acss[ii,:], label=str(gs[ii]),c=c)
plt.legend(fontsize=30)    
plt.xlabel(r'$\tau$',fontsize=40)
plt.ylabel(r'$<s(t)s(t-\tau)>$',fontsize=40)
