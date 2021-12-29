# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:31:47 2020

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

# %% FORCE with GLM-net
###############################################################################
###############################################################################
# %%
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

def NL(x,spkM):
    """
    Passing x through logistic nonlinearity with spkM maximum
    """
    nl = spkM/(1+np.exp(-2*x))
    #nl = np.tanh(x)
    return nl

def deNL(x):
    de = spkM*x*1/np.cosh(x)**2  #derivative of tanh
    #de = x*2/(1+np.exp(-2*x))**2 * np.exp(-2*x)
    return de

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
#    N = len(x)
#    spike = np.random.rand(N) < x*dt*0.1
    x[x<0] = 0
    #x[x>100] = 100
    spike = np.random.poisson(x*dt)  #Poisson process
#    spike = x*dt #rate model
    return spike

# %% setup
#size and length
N = 300
T = 200
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
p_glm = .2
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = 1.
E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
nbasis = 5
pad = 100
spkM = 1
tau = 1
thetas = np.random.randn(N,N,nbasis)/1  #tensor of kernel weights
M_ = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1

for ii in range(N) :
    jj = np.where(np.abs(M_[ii,:])>0)
    M_[ii,jj] = M_[ii,jj] - np.sum(M_[ii,jj])/len(jj)
M_ = M_ * mask

Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((N,N,pad))  #number of kernels x length of time window
sparse = np.random.rand(N,N)
#mask = np.random.rand(N,N)
#mask[sparse>p_glm] = 0
#mask[sparse<=p_glm] = 1
for ii in range(N):
    for jj in range(N):
        temp = np.dot(thetas[ii,jj,:], Ks)
        if ii==jj:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*1.
        else:
            #temp = temp - np.mean(temp)
            #temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = 0#temp*mask[ii,jj]  #0 for non-diagonal

#input parameters
wo = np.ones((N,1))
dw = np.zeros((N,1))
wf_w = 2.0*(np.random.randn(N,nbasis)-0.5)
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 4
ft = 1*(amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    1*(amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    1*(amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft[ft<0] = 0
ft = ft*100#/1.5
#ft = Xt[:,0]*10  #test for chaos
#ft = target*10 #test for IO target
#stim = stim_*0
ft = np.concatenate((np.zeros(pad),ft))

#initial conditions
wo_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,pad))
rt = np.zeros((N,pad))
spks = np.zeros((N,pad))

xt[:,0] = x0
rt[:,0] = NL(xt[:,0],spkM)
z = z0

# %% learning
P = (1.0/alpha)*np.eye(N)
for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    tens = NL( np.einsum('ijk,jk->i',  allK, spks), spkM) 
    spks_ = spiking( (M_ @ tens +0) , dt) #+ stim[:,tt]
    spks_[spks_>0] = 1
    spks = np.concatenate((spks[:,1:],spks_[:,None]),axis=1)  #time iterative update
    rt = tens
    
    #reconstruct dynamics
    z = wo.T @ tens
    
    #learning
    if np.mod(tt, learn_every) == 0:
        dr = (tens)  #taking derivative over the nonlinearity #deNL
        k = (P @ dr)[:,None]
        rPr = dr[:,None].T @ k
        c = 1.0/(1.0 + rPr)
        P = P - (k @ k.T) * c  #projection matrix
    	
        # update the error for the linear readout
        e = z-ft[tt] ### how is error computed!
	
    	# update the output weights
        dw = -(e[:,None]*k*c)#[:,None]
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M_ = M_ + np.repeat(dw,N,1).T
     
    #print(tt,z)
    # Store the output of the system.
    zt[tt] = np.squeeze(z)
    wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))
    
# %% plotting
plt.figure()
error_avg = sum(abs(zt-ft[:-pad]))/simtime_len
print(['Training MAE: ', str(error_avg)]) 
print('mean firing:', np.mean(spks))  
print(['Now testing... please wait.'])

plt.subplot(211)
plt.plot(ft[:-pad])
plt.plot(zt,'--')
plt.subplot(212)
plt.plot(wo_len)

plt.figure()
plt.imshow(spks,aspect='auto')

# %% testing
zpt = np.zeros(len(simtime))
M0 = np.random.randn(N,N)*g*scale
spks = np.zeros((N,pad))

for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    tens = NL( np.einsum('ijk,jk->i',  allK, spks), spkM)
    spks_ = spiking( (M_ @ tens +0) , dt) #+ stim[:,tt]
    spks_[spks_>0] = 1
    spks = np.concatenate((spks[:,1:],spks_[:,None]),axis=1)  #time iterative update
    rt = tens
  
    #reconstruct dynamics
    z = wo.T @ tens
    zpt[tt] = z

plt.figure()
plt.plot(ft[:])
plt.plot(zpt*1)

plt.figure()
plt.imshow(spks, aspect='auto')
error_avg = sum(abs(zpt-ft[:-pad]))/simtime_len
print(['Training MAE: ', str(error_avg)]) 
print('mean firing:', np.mean(spks))

# %% analysis
plt.figure()
plt.plot(Rs_tr,MSE_tr,'o',markersize=20,label='test')
plt.plot(Rs_te,MSE_te,'o',markersize=20,label='train')
plt.xlabel('mean Poisson rate',fontsize=40)
plt.ylabel('MSE',fontsize=40)
plt.legend(fontsize=30)

# %% Target dynamics
###############################################################################
# %% moving average
npulse = 10
imp_a = np.random.randn(npulse)*10
imp_dur = 30
tau = 5
time_points = np.random.choice(simtime_len,npulse)
temp = np.zeros(simtime_len)
for ii,tt in enumerate(time_points):
    temp[tt:tt+imp_dur] = imp_a[ii]
#input_ = np.array([temp[tt:tt+imp_dur] = imp_a[ii] for ii,tt in enumerate(time_points)])
input_ = temp
target = np.zeros(simtime_len)
cnt = 0
for tt in range(simtime_len-1):
    target[tt+1] = target[tt] + dt/tau*(input_[tt])

plt.figure()
plt.plot(input_)
plt.plot(target)

# %%
impulse = np.random.randint(-1,2,N)
stim_ = np.zeros((N,simtime_len))
for ii,tt in enumerate(time_points):
    stim_[:,tt:tt+imp_dur] = np.repeat(impulse[:,None]*imp_a[ii],imp_dur,axis=1)

# %% 3-bit (with higher dimension)


