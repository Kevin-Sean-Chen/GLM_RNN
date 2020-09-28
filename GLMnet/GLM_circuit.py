# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:11:04 2020

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
# %% neural circuit dynamics model
###settings
N = 3  #number of neurons
dt = 0.1  #ms
T = 10000  #total time
time = np.arange(0,T,dt)   #time axis
lt = len(time)

x = np.zeros((N,lt))  #voltage
spk = np.zeros_like(x)  #spikes
syn = np.zeros_like(x)  #synaptic efficacy
rate = np.zeros_like(x)  #spike rate
x[:,0] = np.random.randn(N)*1
syn[:,0] = np.random.rand(N)
J = np.array([[6.8, -2.5, -2],\
              [-3, 7., -2],\
              [-2.3, -2.5, 4.1]])
J = J.T*2.  #connectivity matrix
noise = 1.  #noise strength
stim = np.random.randn(lt)*20  #np.random.randn(N,lt)*20.  #common random stimuli
taum = 5  #5 ms
taus = 50  #50 ms
E = 1

eps = 10**-15
def LN(x):
    """
    logistic nonlinearity
    """
    ln = 1/(1+np.exp(-x*1.+eps))   #logsitic
#    ln = np.array([max(min(100,xx),0) for xx in x])  #ReLu
#    ln = np.log(1+np.exp(x))  #sloft-max
    return np.random.poisson(ln) #ln  #Poinsson emission

def spiking(ll,dt):
    """
    Given Poisson rate (spk per second) and time steps dt return binary process with the probability
    """
    N = len(ll)
    spike = np.random.rand(N) < ll*dt  #for Bernouli process
    return spike

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[tt]*np.array([1,1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
    spk[:,tt+1] = spiking(LN(x[:,tt+1]),dt)
    rate[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = 1#syn[:,tt] + dt*( (1-syn[:,tt])/taus - syn[:,tt]*E*spk[:,tt] )

### plotting 
plt.figure()
plt.subplot(411)
plt.imshow(x, aspect='auto');
plt.subplot(412)
plt.imshow(spk, aspect='auto');
plt.subplot(413)
plt.imshow(rate, aspect='auto');
plt.xlim([0,time[-1]])
plt.subplot(414)
plt.plot(time, rate.T);
plt.xlim([0,time[-1]])

# %%
###############################################################################
# %% functions for GLM inference
eps = 10**-15

def neglog(theta, Y, X, pad, nb):
    """
    negative log-likelihood to optimize theta (parameters for kernel of all neurons)
    with neural responses time length T x N neurons, and the padding window size
    return the neg-ll value to be minimized
    """
    k = kernel(theta,pad)
    v = LN(X @ k)  #nonlinear function
    nl_each = -(np.matmul(Y.T, np.log(v+eps)) - np.sum(v))  #Poisson negative log-likelihood
    #nl_each = -( np.matmul(Y.T, np.log(v+eps)) - np.matmul( (1-Y).T, np.log(1-v+eps)) )  #Bernouli process of binary spikes
    nl = nl_each#.sum()
    return nl

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

def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
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
    fkt = beta*(tl/k)**n*np.exp(-n*(tl/k))
    return fkt

def build_matrix(stimulus, spikes, pad, couple):
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
    if couple==0:
        X = X.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
    elif couple==1:
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
        X = X_s_h.copy()
#        #print(hh)
    
    return X

def build_convolved_matrix(stimulus, spikes, Ks, couple):
    """
    Given stimulus and spikes, construct design matrix with features being the value projected onto kernels in Ks
    stimulus: Tx1
    spikes: TxN
    Ks: kxpad (k kernels with time window pad)
    couple: binary option with (1) or without (0) coupling
    """
    T, N = spikes.shape
    k, pad = Ks.shape
    
    Stimpad = np.concatenate((stimulus,np.zeros((pad,1))),axis=0)
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    Xstim = np.squeeze(Stimpad[S])
    Xstim_proj = np.array([Xstim @ Ks[kk,:] for kk in range(k)]).T
    
    if couple==0:
        X = np.concatenate((np.ones((T,1)), Xstim_proj),axis=1)
    elif couple==1:
        spkpad = np.concatenate((spikes,np.zeros((pad,N))),axis=0)
        Xhist = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
        Xhist_proj = [np.array([Xhist[nn] @ Ks[kk,:] for kk in range (k)]).T for nn in range(N)]
        
        X = Xstim_proj.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
        for hh in range(0,N):
            X = np.concatenate((X,Xhist_proj[hh]),axis=1)
    return X


# %%
###############################################################################
# %% inference method (single)
nneuron = 2
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
Y = np.squeeze(rate[nneuron,:])  #spike train of interest
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
stimulus = stim[:,None]  #same stimulus for all neurons
X = build_convolved_matrix(stimulus, rate.T, Ks, couple)  #design matrix with features projected onto basis functions
###pyGLMnet function with optimal parameters
glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)

# %% direct simulation
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
plt.figure()
plt.plot(Y*1.)  #ground truth
plt.plot(yhat,'--')

# %%reconstruct kernel
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

# %%
# %% inference method (all-together)
allK_rec = np.zeros((N,N+1,pad))
all_yhat = np.zeros((N,lt))
X_bas_n = build_convolved_matrix(stim[:,None], rate.T, Ks, couple)  #design matrix projected to basis functions
for nn in range(N):
    Yn = rate[nn,:]  #nn-th neuron
    glm = GLMCV(distr="binomial", tol=1e-5,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)
    glm.fit(X_bas_n, Yn)
    ###store kernel
    theta_rec_n = glm.beta_[1:]
    theta_rec_n = theta_rec_n.reshape(N+1,nbasis)
    for kk in range(N+1):
        allK_rec[nn,kk,:] = np.dot(theta_rec_n[kk,:], Ks)   #reconstructing all kernels
    ###store prediction
    all_yhat[nn,:] = simulate_glm('binomial', glm.beta0_, glm.beta_, X_bas_n)
    
# %% reconstruct output
plt.figure()
plt.subplot(211)
plt.imshow(rate, aspect='auto')
plt.subplot(212)
plt.imshow(all_yhat,aspect='auto')

# %% reconstruct kernels
nneuron = 1
K_rec_norm = np.array([allK_rec[nneuron,ii,:]/np.linalg.norm(allK_rec[nneuron,ii,:]) for ii in range(N+1)])
plt.figure()
plt.plot(K_rec_norm.T)

# %% reconstruct all kernels
plt.figure()
plt.plot(np.squeeze(allK_rec[:,0,:]).T)
plt.figure()
kk = 0
for jj in range(0,N):
    for ii in range(0,N):
        plt.subplot(3,3,kk+1)
        plt.plot(allK_rec[jj,ii+1,:])
        kk = kk+1

# %% Scaling
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def sim_circuit(lt):
    """
    Exact same model as above but put in function for to scan through different lengths
    """
    x = np.zeros((N,lt))  #voltage
    spk = np.zeros_like(x)  #spikes
    rate = np.zeros_like(x)  #spike rate
    x[:,0] = np.random.randn(N)*1
    stim = np.random.randn(lt)*20  #np.random.randn(N,lt)*20.  #common random stimuli
    
    ###iterations for neural dynamics
    for tt in range(0,lt-1):
        x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,LN(1*x[:,tt]))) + stim[tt]*np.array([1,1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
        spk[:,tt+1] = spiking(LN(x[:,tt+1]),dt)
        rate[:,tt+1] = LN(x[:,tt+1])
    return rate, stim

def comp_varexp(Y,S,nneuron):
    """
    Compute variance explained from the GLM results, with parameters set above
    """
    y = np.squeeze(Y[nneuron,:])  #spike train of interest
    stimulus = S[:,None]  #same stimulus for all neurons
    X = build_convolved_matrix(stimulus, Y.T, Ks, couple)  #design matrix with features projected onto basis functions
    glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
                score_metric="deviance",
                alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
    glm.fit(X, y)
    yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
    varexp = np.corrcoef(y,yhat)
    return varexp

lts = np.array([200,500,1000,5000,10000,20000])
VARS = np.zeros((N,len(lts)))
Y, S = sim_circuit(20000)
for nn in range(N):
    for ti,tt in enumerate(lts):
#        Y, S = sim_circuit(tt)
        varexp = comp_varexp(Y[:,:tt],S[:tt],nn)
        VARS[nn,ti] = varexp[0][1]

# %%
plt.figure()
plt.plot(lts,VARS.T,'-o')
plt.xlabel('length of simulation')
plt.ylabel('variance explained')

