#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:44:01 2020

@author: kschen
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

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

# %% NEURAL DYNAMICS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
taum = 5  #fast time scale
taus = 10  #slow time scale
E = 1.  #spiking adaptation strength

eps = 10**-15
def LN(x):
    """
    Logistic nonlinearity
    """
    return np.random.poisson(100/(1+np.exp(-x*1.+eps)))

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt*( -x[:,tt]/taum + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[tt]*np.array([1,1,1,1]) + noise*np.random.randn(4)*np.sqrt(dt))
    spk[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = syn[:,tt] + dt*( (1-syn[:,tt])/taus - spk[:,tt]*E )
    
plt.figure() 
plt.subplot(211)
plt.imshow(spk,aspect='auto');
plt.subplot(212)
plt.plot(time,x.T);

# %% three-neuron random circuit
###settings
N = 3
dt = 0.1  #ms
T = 2000
time = np.arange(0,T,dt)
lt = len(time)

x = np.zeros((N,lt))  #voltage
spk = np.zeros_like(x)  #spikes
syn = np.zeros_like(x)  #synaptic efficacy
x[:,0] = np.random.randn(N)*1
#J = np.random.randn(N,N)
J = np.array([[2.3, 0.95, -0.3],\
              [-1.6, -0.05, -1.6],\
              [-0.3,1.2, 2.1]])
#J = np.array([[1.4, -0.3, 0.8],\
#              [-0.3, 1.4, 0.8],\
#              [-2.5,-2.5, -1]])
J = J.T*1.
noise = .1
stim = np.random.randn(N,lt)*50. #np.random.randn(lt)*1.
#stim[0,1:1001] = np.random.randn(1,1000)*1 + 40
#stim[2,5000:6000] = np.random.randn(1,1000)*1 + 20
#stim[0,10000:11000] = np.random.randn(1,1000)*1 + 40
taum = 1
taus = 50
E = 1.

eps = 10**-15
def LN(x):
    """
    nonlinearity
    """
    ln = np.random.poisson(50/(1+np.exp(-x*1.+eps)))
#    ln = np.array([max(min(100,xx),0) for xx in x])
    return ln

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[0,tt]*np.array([1,1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
    spk[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = 1#syn[:,tt] + dt*( (1-syn[:,tt])/taus - spk[:,tt]*E )
    
plt.figure()
plt.subplot(311)
plt.imshow(spk,aspect='auto');
plt.subplot(312)
plt.plot(time,x.T);
plt.subplot(313)
plt.plot(stim.T);

# %%
plt.figure()
plt.subplot(211)
plt.imshow(spk,aspect='auto',extent=[0,T*dt,0,N])
plt.subplot(212)
plt.plot(time, x.T)
plt.xlabel('time (ms)')
plt.ylabel('potential V')

# %% INFERENCE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% let's fit GLM-net!!!
def neglog(theta, Y, X, pad, nb):
    """
    negative log-likelihood to optimize theta (parameters for kernel of all neurons)
    with neural responses time length T x N neurons, and the padding window size
    return the neg-ll value to be minimized
    """
    T,N = Y.shape
    dc = theta[:N]  #the DC compontent (baseline fiting) that's not dynamic-dependent
    wo_dc = theta[N:]  #parameters to construct kernels
    pars_per_cell = int(len(wo_dc)/(N))  #the number of parameters per neurons (assuming they're the same)
    kernels_per_cell = int(pars_per_cell/nb)  #the number of kernels per neuron (assuming they're the same)
    theta_each = np.reshape(wo_dc, (N, kernels_per_cell, nb))  #N x kernels x nb
    k = np.array([kernel(theta_each[nn,kk,:], pad) for nn in range(0,N) for kk in range(0,kernels_per_cell)])  #build kernel for each neuron
    k = k.reshape((N, kernels_per_cell*pad))  # N x (kernel_per_cell*pad)
    k =  np.concatenate((dc[:,None], k),axis=1).T # adding back the DC baseline
    v = LN(np.matmul(X,k))  #nonlinear function
    nl_each = -(np.matmul(Y.T, np.log(v+eps)) - np.sum(v))  #Poisson negative log-likelihood
    nl = nl_each.sum()
    return nl

def kernel(theta, pad):
    """
    Given theta weights and the time window for padding,
    return the kernel contructed with basis function
    """
    nb = len(theta)
    basis = basis_function1(pad, nb)  #construct basises
    k = np.dot(theta, basis.T)  #construct kernels with parameter-weighted sum
    return k

def basis_function1(nkbins, nBases):
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

def basis_function2(n, k, tl):
    """
    More biophysical delayed function, given a width parameter n, location of kernel k,
    and the time window tl (n=5-10 is a normal choice)
    """
    beta = np.exp(n)
    fkt = beta*(tl/k)**n*np.exp(-n*(tl/k))
    return fkt

def build_matrix(stimulus, spikes, pad):
    """
    Given time series stimulus (T time x N neurons) and spikes of the same dimension and pad length,
    build and return the design matrix with stimulus history, spike history od itself and other neurons
    """
    T, N = spikes.shape  #neurons and time
    SN = stimulus.shape[0]  #if neurons have different input (ignore this for now)
    
    # Extend Stim with a padding of zeros
    Stimpad = np.concatenate((stimulus,np.zeros((pad,1))),axis=0)
    # Broadcast a sampling matrix to sample Stim
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    X = np.squeeze(Stimpad[S])
    X_stim = np.concatenate((np.ones((T,1)), X),axis=1)  #for DC component that models baseline firing
#    h = np.arange(1, 6)
#    padding = np.zeros(h.shape[0] - 1, h.dtype)
#    first_col = np.r_[h, padding]
#    first_row = np.r_[h[0], padding]
#    H = linalg.toeplitz(first_col, first_row)
    
    # Spiking history and coupling
    spkpad = np.concatenate((spikes,np.zeros((pad,N))),axis=0)
    # Broadcast a sampling matrix to sample Stim
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    X_h = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
    # Concatenate the neuron's history with old design matrix
    X_s_h = X_stim.copy()
    for hh in range(0,N):
        X_s_h = np.concatenate((X_s_h,X_h[hh]),axis=1)
        #print(hh)
    
    return X_s_h

# %% main analysis part
pad = 40  #window size of the kernel
nbasis = 7 #number of basis use for kernel fitting
Y = spk.T.copy()
X = build_matrix(stim[0,:][:,None], Y, pad)
T, N = spk.T.shape
npars = (N**2+N)*nbasis+N  # number of paramerers for all kernels would be (NxN+N)*nb+N due to interaction and receptive field and their baseline (assuming they're the same size)
theta0 = np.random.randn(npars)
res = sp.optimize.minimize( neglog, theta0, args=(Y, X, pad, nbasis), method="L-BFGS-B", tol=1e-3, options={'disp':True,'gtol':1e-2})#

# %% simulation with GLM-net!

### un-pack the inferred kernels
theta_infer = res.x
dcs = theta_infer[:N] #the DC compontent (baseline fiting) that's not dynamic-dependent
wo_dc = theta_infer[N:]  #parameters to construct kernels  
pars_per_cell = int(len(wo_dc)/(N))  #the number of parameters per neurons (assuming they're the same)
kernels_per_cell = int(pars_per_cell/nbasis)  #the number of kernels per neuron (assuming they're the same)
theta_each = np.reshape(wo_dc, (N, kernels_per_cell, nbasis))  #N x kernels x nb
allK = np.array([kernel(theta_each[nn,kk,:], pad) for nn in range(0,N) for kk in range(0,kernels_per_cell)])  #build kernel for each neuron
allK = allK.reshape((N, kernels_per_cell, pad))  # N x (kernel_per_cell*pad)

# %%
### GLM network simulation
def GLM_net(allK, dcs, S):
    """
    Simulate a GLM network given all response and coupling kernels and the stimulus
    """
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    _,T = S.shape  #all the same stimulus for now
    us = np.zeros((N,T))  #all activity through time
    K_stim = allK[:,0,:]  # N x pad response kernels
    K_couple = allK[:,1:,:]  # N x N xpad coupling filter between neurons (and itself)
    #K_couple.transpose(1, 0, 2).strides
    
    for tt in range(h,T):
        ut = np.einsum('ij,ij->i', S[:,tt-h:tt], K_stim) + \
             np.einsum('ijk,jk->i',  K_couple, us[:,tt-h:tt])  #neat way for linear dynamics
        ut = LN(ut + dcs)  #share the same nonlinearity for now
        us[:,tt] = np.random.poisson(ut)
    return us
TT = 2000#lt
S = np.repeat(stim[0,:TT][:,None],3,axis=1).T #stim.copy()
#S = np.random.randn(3,T)
us_sim = GLM_net(allK, dcs, S)
plt.figure()
plt.subplot(211)
plt.imshow(us_sim,aspect='auto',extent=[0,TT*dt,0,N])
plt.subplot(212)
plt.plot(time[:TT], us_sim.T)
plt.xlabel('time (ms)')
plt.ylabel('firing rate')

# %%
###check kernels
plt.figure()
kk = 0
for jj in range(0,3):
    for ii in range(0,3):
        plt.subplot(3,3,kk+1)
        plt.plot(allK[ii,jj+1,:])
        kk = kk+1