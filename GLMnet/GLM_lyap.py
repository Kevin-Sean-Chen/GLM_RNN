# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:52:50 2021

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
#    spike = x
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
        temp = spiking(NL(ut, spkM), dt)
        temp[temp>0]=1
        spks[:,tt] = temp
#        spks[:,tt] = spiking(NL(ut, spkM), dt)  #Bernouli process for spiking
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
        temp = spiking(M_ @ NL(r, spkM), dt)
#        temp[temp>0]=1
        spks[:,tt] = temp
#        spks[:,tt] = spiking(M_ @ NL(r, spkM), dt)  #Bernouli process for spiking
    return us, spks

# %%
# %% GLM kernel setup
N = 100
nbasis = 5
pad = 100
spkM = 1
T = 2000
dt = 1
p_glm = 0.5
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
            allK[ii,jj,:] = temp*1.
#            allK[ii,jj,:] = temp*mask[ii,jj]
        else:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*mask[ii,jj]
            
# %% Lyapunov exp
def GLM_net_pert(M_, allK, T, pert):
    N, K, h = allK.shape  #N neurons x K kernels x pad window
    us = np.zeros((N,T))  #all activity through time
    spks = np.zeros((N,T))  #for spiking process
    us[:,h+1] = pert/N
    
    for tt in range(h,T):
        ### G-FORCE network
        r = np.einsum('ijk,jk->i',  allK, spks[:,tt-h:tt])  #neat way for linear dynamics
        us[:,tt] = NL(r, spkM)
        temp = spiking(M_ @ NL(r, spkM), dt)
#        temp[temp>0]=1
        spks[:,tt] = temp
        ### GLM network
#        ut = np.einsum('ijk,jk->i',  allK, spks[:,tt-h:tt])  #neat way for linear dynamics
#        us[:,tt] = NL(ut, spkM) #np.random.poisson(ut)
#        temp = spiking(NL(ut, spkM), dt)
#        temp[temp>0]=1
#        spks[:,tt] = temp
    return us, spks
   
spkM = 1
eps = 1e-5
Tt = 500
rg = 0.1
M_ = rg/np.sqrt(p_glm*N)*np.random.randn(N,N)

us, spks = GLM_net_pert(M_, allK, Tt, 0)
us_, spks_ = GLM_net_pert(M_, allK, Tt, eps)

Lya = 1/Tt * np.log( (np.linalg.norm(us[:,-1] - us_[:,-1])+1e-10) / eps ) #-np.log(1e-10/eps)/Tt
print(Lya)

# %%
log_r = np.array([-5,-4,-3,-2,-1,0,1,2,3])
lyas = np.zeros(len(log_r))
for ll in range(len(log_r)):
    MM = np.float_power(10, log_r[ll]) * M_
    us, spks = GLM_net_pert(MM, allK, Tt, 0)
    us_, spks_ = GLM_net_pert(MM, allK, Tt, eps)
    Lya = 1/Tt * np.log( (np.linalg.norm(us[:,-1] - us_[:,-1])+1e-10) / eps )
    lyas[ll] = Lya
# %%
plt.figure()
plt.semilogx(np.float_power(10,log_r), lyas,'-o')
plt.xlabel('recurrent strength g', fontsize=40)
plt.ylabel('Lyapunov exponent', fontsize=40)

###############################################################################
# %%
###############################################################################
# %% Lyapunov from time series
nn = 5
rg = 10
M_ = rg/np.sqrt(p_glm*N)*np.random.randn(N,N)
M_[np.random.rand(N,N)>p_glm] = 0
us,spks = G_FORCE_network(M_, spkM, allK, T)
negative_control = np.sin(np.arange(0,T)/30) + np.random.randn(T)  #stochastic not chaotic

data = us[nn,:].copy()
#data = negative_control.copy()
eps = 1e-3
lyapunovs = [[] for i in range(T)]

maxit = 1000
n = 0 #number of nearby pairs found
for i in range(T):
    for j in range(i+1,T):
        if np.abs(data[i] - data[j]) < eps:
            n+=1
            print(n)
            for k in range(min(T-i,T-j)):
                lyapunovs[k].append(np.log(np.abs(data[i+k] - data[j+k])))
    if n > maxit:
        break

plt.figure()
plt.plot(data)
# %%
#f=open('lyapunov.txt','w')
#for i in range(len(lyapunovs)):
#    if len(lyapunovs[i]):
#        print>>f, i, sum(lyapunovs[i])/len(lyapunovs[i])
#f.close()
      
lya = np.zeros(len(lyapunovs))          
for i in range(len(lyapunovs)):
    if len(lyapunovs[i]):
        temp = lyapunovs[i]
        pos = np.isinf(temp)  #the infinity ones
        mask = np.array(~pos)  #mask them out
        lya[i] = np.nansum(lyapunovs[i]*mask)/(len(lyapunovs[i])-sum(pos))
        
# %%
plt.figure()
plt.plot(lya,'-o')
plt.plot([0,len(lya)],[0,0],'--')

# %%
ggs = np.array([0.001, 0.01, 0.1, 1, 10])
Lyas = np.array([-0.13, 0.03, 0.16, 0.2, .3])
spk_rates = np.array([ 0.0002, 0.002, 0.05, 0.26, 2.2])
plt.figure()
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.semilogx(ggs, Lyas, '-o',linewidth=8, markersize=13)
ax2.semilogx(ggs, spk_rates, '-o', color='g',linewidth=8, markersize=13)

ax1.set_xlabel(r'$g$',fontsize=50)
ax1.set_ylabel(r'$\lambda_{max}$',fontsize=50,color='b')
ax2.set_ylabel('mean spike rate',fontsize=50,color='g')
