# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:22:13 2020

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

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 


from pyglmnet import GLM, simulate_glm
from pyglmnet import GLMCV
from pyglmnet import GLM

#%matplotlib qt5

# %% functions
eps = 10**-15
    
def NL(x):
    """
    Poinsson emission with logistic nonlinearity
    """
    nl = np.random.poisson(10/(1+np.exp(-x)))
#    nl = np.random.poisson(np.log(1+np.exp(x)+eps))
#    nl = np.array([max(min(100,xx),0) for xx in x])  #ReLu
    return nl

def neglog(theta, Y, X, pad, nb):
    """
    negative log-likelihood to optimize theta (parameters for kernel of all neurons)
    with neural responses time length T x N neurons, and the padding window size
    return the neg-ll value to be minimized
    """
#    T,N = Y.shape
#    dc = theta[:N]  #the DC compontent (baseline fiting) that's not dynamic-dependent
#    wo_dc = theta[N:]  #parameters to construct kernels
#    pars_per_cell = int(len(wo_dc)/(N))  #the number of parameters per neurons (assuming they're the same)
#    kernels_per_cell = int(pars_per_cell/nb)  #the number of kernels per neuron (assuming they're the same)
#    theta_each = np.reshape(wo_dc, (N, kernels_per_cell, nb))  #N x kernels x nb
#    k = np.array([kernel(theta_each[nn,kk,:], pad) for nn in range(0,N) for kk in range(0,kernels_per_cell)])  #build kernel for each neuron
#    k = k.reshape((N, kernels_per_cell*pad))  # N x (kernel_per_cell*pad)
#    k =  np.concatenate((dc[:,None], k),axis=1).T # adding back the DC baseline
    
    k = kernel(theta,pad)
    v = NL(X @ k)  #nonlinear function
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
    tl = np.arange(tl)
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

# %% ground truth GLM
T = 1000
stim = np.random.randn(T,1)*1
#stim[stim>0]=1
#stim[stim<=0] = 0
nb = 5
ks = np.random.randn(nb)
pad = 30
K = kernel(ks,pad)[:,None]
spk = NL(np.convolve(stim[:,0],K[:,0],'same'))[:,None]
X = build_matrix(stim,spk,pad,0)[:,1:]
spk = NL(X @ K)

plt.figure()
plt.subplot(121)
plt.plot(K)
plt.subplot(122)
plt.plot(spk)

# %% inference
X = build_matrix(stim,spk,pad,0)[:,1:]
npars = nb
theta0 = 1*np.random.randn(npars)  #np.ones(npars) #ks + 
res = sp.optimize.minimize( neglog, theta0, args=(spk, X, pad, nb), method='Nelder-Mead',options={'disp':True,'maxiter':1000})#, method="L-BFGS-B", tol=1e-10)#, options={'disp':True,'gtol':1e-2})
theta_infer = res.x

# %%
plt.figure()
plt.plot(K)
plt.plot(kernel(theta_infer,pad),'-o')
plt.plot(kernel(theta0,pad),'--')

# %%
###############################################################################
###############################################################################
# %% test pyGLM
### This is super-easy if we rely on built-in GLM fitting code
glm = GLMCV(distr="binomial", tol=1e-3,
            score_metric="pseudo_R2",
            alpha=1.0, learning_rate=3, max_iter=100, cv=3, verbose=True)

glm.fit(X, np.squeeze(spk))

# %% 
plt.figure()
pyglm_infer = glm.beta_
plt.plot(pyglm_infer/np.linalg.norm(pyglm_infer))
plt.plot(K/np.linalg.norm(K),'--')

# %% Two-neuron circuit with pyGLMnet
###############################################################################
###############################################################################
# %% combine basis function here
###convolve with basis first??
N = 2
dt = 0.1  #ms
T = 10000
time = np.arange(0,T,dt)
lt = len(time)

x = np.zeros((N,lt))  #voltage
spk = np.zeros_like(x)  #spikes
syn = np.zeros_like(x)  #synaptic efficacy
syn[:,0] = np.random.rand(N)
rate = np.zeros_like(x)  #spike rate
x[:,0] = np.random.randn(N)*1
#J = np.random.randn(N,N)
J = np.array([[2.1, -1.5],\
              [-1.5, 1.5]])
#J = np.array([[1, -0],\
#          [-0, 1.]])
J = J.T*5.
noise = 0.
stim = np.random.randn(lt)*20-0.  #np.random.randn(N,lt)*20.
taum = 2  #5 ms
taus = 5  #50 ms
E = 1

eps = 10**-15
def LN(x):
    """
    nonlinearity
    """
    ln = 1/(1+np.exp(-x*1.+eps))   #logsitic
#    ln = np.log(1+np.exp(x)+eps)
#    ln = np.array([max(min(100,xx),0) for xx in x])  #ReLu
    return ln  #np.random.poisson(ln) #ln  #Poinsson emission

def spiking(ll,dt):
    """
    Given Poisson rate (spk per second) and time steps dt return binary process with the probability
    """
    N = len(ll)
    spike = np.random.rand(N) < ll*dt  #for Bernouli process
    return spike

###iterations for neural dynamics
for tt in range(0,lt-1):
    x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,LN(syn[:,tt]*x[:,tt]))) + stim[tt]*np.array([1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
    spk[:,tt+1] = spiking(LN(x[:,tt+1]),dt)
    rate[:,tt+1] = LN(x[:,tt+1])
    syn[:,tt+1] = 1 #syn[:,tt] + dt*( (1-syn[:,tt])/taus - syn[:,tt]*E*rate[:,tt] )
    
plt.figure()
plt.subplot(411)
plt.imshow(spk,aspect='auto');
plt.subplot(412)
plt.imshow(rate,aspect='auto');
plt.subplot(413)
plt.plot(time,x.T);
plt.xlim([0,time[-1]])
plt.subplot(414)
plt.plot(time,stim.T);
plt.xlim([0,time[-1]])

plt.figure()
plt.plot(rate[0,:])
# %% raw points
pad = 50
X = build_matrix(stim[:,None], spk.T, pad, 1)
glm = GLMCV(distr="poisson", tol=1e-5, eta=2.0,
            score_metric="deviance",
            alpha=0., learning_rate=0.01, max_iter=1000, cv=3, verbose=True)
glm.fit(X, np.squeeze(rate[0,:]))

# %%
yhat = simulate_glm('poisson', glm.beta0_, glm.beta_, X)
plt.plot(rate[0,:],'--')
plt.plot(yhat)


# %%
#reg_lambda = np.logspace(np.log(1e-6), np.log(1e-8), 100, base=np.exp(1))
#glm_poissonexp = GLMCV(distr='poisson', verbose=False, alpha=0.05,
#            max_iter=100, learning_rate=2e-1, score_metric='pseudo_R2',
#            reg_lambda=reg_lambda, eta=4.0)
#glm_poissonexp.fit(X, np.squeeze(spk[0,:]))

# %%
# %%
# %% test with basis function
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1
Y = np.squeeze(rate[0,:])  #spike train of interest
#Ks = basis_function1(pad, nbasis).T  #basis function for projection
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
stimulus = stim[:,None]
X = build_convolved_matrix(stimulus, rate.T, Ks, couple)
#glm = GLMCV(distr="poisson", tol=1e-5,
#            score_metric="pseudo_R2",
#            alpha=0., learning_rate=0.01, max_iter=1000, cv=3, verbose=True)
glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)


# %% direct simulation
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)
plt.figure()
#plt.subplot(211)
#plt.plot(yhat)
#plt.subplot(212)
plt.plot(yhat)#(spiking(LN(yhat),dt))
plt.plot(Y*1.,'--')

plt.figure()
Y_bin = Y.copy()
Y_bin[Y>1] = 1
yhat_bin = yhat.copy()
yhat_bin[yhat_bin<1] = 0
plt.imshow(np.concatenate((Y_bin[:,None], yhat_bin[:,None]), axis=1).T, aspect='auto')

# %% unpacking basis
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

# %% reconstruction
rec_Ks = allKs.reshape(-1)+1
X_rec = build_matrix(stimulus, spk.T, pad, couple)[:,1:]
conv_ = X_rec @ rec_Ks
spk_rec = spiking(LN(conv_+dc_-0),dt)

plt.figure()
plt.plot(Y)
plt.plot(spk_rec,'--')

plt.figure()
plt.imshow(np.concatenate((Y_bin[:,None], spk_rec[:,None]), axis=1).T, aspect='auto')

# %% pyGLMnet example
n_samples, n_features = 1000, 100
distr = 'poisson'

# random sparse regressors
beta0 = np.random.rand()
beta = sp.sparse.random(1, n_features, density=0.2).toarray()[0]
# simulate data
Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
ytrain = simulate_glm('poisson', beta0, beta, Xtrain)
Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
ytest = simulate_glm('poisson', beta0, beta, Xtest)

# create an instance of the GLM class
glm = GLM(distr='poisson')

# fit the model on the training data
glm.fit(Xtrain, ytrain)

# predict using fitted model on the test data
yhat = glm.predict(Xtest)

plt.figure()
plt.plot(ytest)
plt.plot(yhat)

# %%
###############################################################################
# Ground truth with kernel-circuit, then infer the kernels
###############################################################################
# %%
### GLM network simulation
def GLM_net(allK, dcs, S):
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
        ut = LN(ut + dcs)  #share the same nonlinearity for now
        us[:,tt] = ut #np.random.poisson(ut)
        spks[:,tt] = spiking(LN(ut + dcs), dt)  #Bernouli process for spiking
    return us, spks

# %% ground truth GLM-net
N = 3
T = 1000
dt = 0.1
time = np.arange(0,T,dt)
stim = np.random.randn(len(time))*0.1
stimulus = np.repeat(stim[:,None],3,axis=1).T #identical stimulus for all three neurons for now
nbasis = 7
pad = 100
nkernels = N**2+N  #xN coupling and N stimulus filters
thetas = np.random.randn(nkernels, nbasis)  #weights on kernels
#Ks = basis_function1(pad, nbasis)
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((nkernels,pad))  #number of kernels x length of time window
for ii in range(nkernels):
    allK[ii,:] = np.dot(thetas[ii,:], Ks)
allK = allK.reshape(N,N+1,pad)
us, spks = GLM_net(allK, 0, stimulus)

plt.figure()
plt.imshow(us,aspect='auto')

# %% inference
nneuron = 0
couple = 1
Y = us[nneuron,:]  #spike train of interest
X_bas = build_convolved_matrix(stim[:,None], spks.T, Ks, couple)  #kernel projected to basis functions
#X_raw = build_matrix(stim[:,None], spks.T, pad, 1)  #directly fitting the whole kernel (with regulaization in GLM)
#glm = GLMCV(distr="poisson", tol=1e-5, eta=1.0,
#            score_metric="deviance",
#            alpha=0., learning_rate=0.01, max_iter=1000, cv=3, verbose=True)
glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=0.1, max_iter=1000, cv=3, verbose=True)
glm.fit(X_bas, Y)

# %% direct simulation
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X_bas)
plt.figure()
#plt.subplot(211)
plt.plot(Y*1.,'--')
plt.plot(yhat)
#plt.subplot(212)
#plt.plot(Y,'--')
#plt.plot(spiking(yhat/dt,dt)*0.5) #(us[nneuron,:])#

# %%reconstruct kernel
theta_rec = glm.beta_[1:]
theta_rec = theta_rec.reshape(N+1,nbasis)
K_rec = np.zeros((N+1,pad))
for ii in range(N+1):
    K_rec[ii,:] = np.dot(theta_rec[ii,:], Ks)
###normalize kernels
K_rec_norm = np.array([K_rec[ii,:]/np.linalg.norm(K_rec[ii,:]) for ii in range(N+1)])
K_tru_norm = np.array([allK[nneuron,ii,:]/np.linalg.norm(allK[nneuron,ii,:]) for ii in range(N+1)])
plt.figure()
plt.subplot(211)
plt.plot(K_rec_norm.T)
plt.subplot(212)
plt.plot(K_tru_norm.T)

plt.figure()
plt.plot(K_rec_norm.T)
plt.plot(K_tru_norm.T,'--')
### array([0.05591984, 0.02842407, 0.00187494, 0.07887918])
# %%
# %% all-together
allK_rec = np.zeros_like(allK)
all_yhat = np.zeros_like(spks)
for nn in range(N):
    Yn = spks[nn,:]
    X_bas_n = build_convolved_matrix(stim[:,None], spks.T, Ks, couple)
    glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=0.01, max_iter=1000, cv=3, verbose=True)
    glm.fit(X_bas, Y)
    ###store kernel
    theta_rec_n = glm.beta_[1:]
    theta_rec_n = theta_rec_n.reshape(N+1,nbasis)
    for kk in range(N+1):
        allK_rec[nn,kk,:] = np.dot(theta_rec_n[kk,:], Ks)
    ###store prediction
    all_yhat[nn,:] = simulate_glm('binomial', glm.beta0_, glm.beta_, X_bas_n)
    
# %% reconstruct output
us_rec, spks_rec = GLM_net(allK_rec, 0, stimulus)
plt.figure()
plt.imshow(spks_rec,aspect='auto')

# %% reconstruct kernels
nneuron = 1
K_rec_norm = np.array([allK_rec[nneuron,ii,:]/np.linalg.norm(allK_rec[nneuron,ii,:]) for ii in range(N+1)])
K_tru_norm = np.array([allK[nneuron,ii,:]/np.linalg.norm(allK[nneuron,ii,:]) for ii in range(N+1)])
plt.figure()
plt.subplot(211)
plt.plot(K_rec_norm.T)
plt.subplot(212)
plt.plot(K_tru_norm.T)

# %% scanning data length
def kernel_MSE(Y,S,nneuron,Ks):
    """
    Compute MSE from the GLM results, with simulated activity, stimulus, and the ground truth kernel
    """
    ###GLM
    y = np.squeeze(Y[nneuron,:])  #spike train of interest
    stimulus = S[:,None]  #same stimulus for all neurons
    X = build_convolved_matrix(stimulus, Y.T, Ks, couple)  #design matrix with features projected onto basis functions
    glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=0.1, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
    glm.fit(X, y)
    
    ###store kernel
    theta_rec = glm.beta_[1:]
    theta_rec = theta_rec.reshape(nbasis,N+1)
    K_rec = np.zeros((N+1,pad))
    for ii in range(N+1):
        K_rec[ii,:] = np.dot(theta_rec[:,ii], Ks)
    
    ###normalize and shift kernels
    K_rec_norm = np.array([K_rec[ii,:]/np.linalg.norm(K_rec[ii,:]) for ii in range(N+1)])  #
    K_tru_norm = np.array([allK[nneuron,ii,:]/np.linalg.norm(allK[nneuron,ii,:]) for ii in range(N+1)])  #
#    K_rec_norm = K_rec_norm.T-K_rec_norm[:,-1]
#    K_tru_norm = K_tru_norm.T-K_tru_norm[:,-1]
    mses = np.sum((K_rec_norm-K_tru_norm)**2,axis=1)  #measuring MSE for each kernel
    return mses

lts = np.array([500,1000,2000,5000,10000,20000])
MSEs = np.zeros((N,N+1,len(lts)))  #MSE for all Nx(N+1) kernels
stim = np.random.randn(len(time))*0.1
stimulus = np.repeat(stim[:,None],3,axis=1).T #identical stimulus for all three neurons for now
us, spks = GLM_net(allK, 0, stimulus)  #total simulation of activity
for nn in range(N):
    for ti,tt in enumerate(lts):
#        Y, S = kernel_MSE(tt)
        MSEs[nn,:,ti] = kernel_MSE(us[:,:tt], stim[:tt], nn, Ks)

# %%
plt.figure()
for ii in range(N):
    normed_mse = MSEs[ii,:,0]  #np.max(MSEs[ii,:,:],axis=1)
    plt.plot((lts[:]),MSEs[ii,ii+1,:].T,'-o')
plt.xlabel('length of simulation')
plt.ylabel('MSE')
