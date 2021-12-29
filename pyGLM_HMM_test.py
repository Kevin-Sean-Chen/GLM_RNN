#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:26:22 2020

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


from pyglmnet import GLM, simulate_glm
from pyglmnet import GLMCV
from pyglmnet import GLM

global eps
eps = 10**-15  #flow-point precision
    
# %% main function
def GLMHMM(y, X, W0, K0, fitopts):
    """
    takes time series of observation y, stimulus X, initial GLM weights W0 and state-transitions T0
    for fitting with firopts for the model (GLM type, iterations, threshold etc.)
    returns result class with tranistion matrix and GLM results
    y: length time T
    X: length time T x w stimulus window (the design matrix)
    W0: l x h (length of kernel x hidden states)
    K0: l x h x h (length of kernel x from j hidden state to k hidden state)
    (T0: h x h)
    """
    ###import dimensions and initialize probabilities
    thresh = fitopts.thresh #10**-3
    T_ = len(y)
    h = W0.shape[1]
    l = X.shape[1]  #might change due to basis function merthod for kernel !!??
    pi0 = np.ones(h)
    pi0 = pi0/sum(pi0)
    
    W = W0.copy()  #weights for GLM eimssion  
    K = K0.copy()  #initial GLM for transitions
#    T = T0.copy()
    niter = fitopts.niter
    lls = np.zeros((h,T_))  #hidden state x time series
    loglik0 = -np.infty

    for ii in range(0,niter):   #iteration loop
        
        #evaluate ll of emission
        etas = np.matmul(X,W)
        for kk in range(h):   #hidden state loop
            lls[kk,:] = evalGLMLL( y, etas[:,kk][:,None], np.ones(1)[:,None], 0, fitopts.family, fitopts.familyextra, 1)
        lls = np.exp(-lls)
        
        ###########
        #compute transition probability from kernels (eqn 26 in Paninski note)
        gKx = np.einsum('ij,jkl->ikl', X, np.einsum('ijk->kij', K))#np.matmul(X, K)  # T x h x h matrix
        Temp = np.zeros((h,h))  #temperary befor normalized
        for jj in range(h):
            for kk in range(h):
                Temp[jj,kk] = evalGLMLL( y, gKx[:,jj,kk][:,None], np.ones(1)[:,None], 0, fitopts.family, fitopts.familyextra, 1)
                ####
                # transition matrix is time dependent!?
                ####
        T = Temp/(1+(Temp-np.diag(Temp)).sum(1))
        np.fill_diagonal(T, 1/(1+(Temp-np.diag(Temp)).sum(1)))
        ###########
        
        ### Forward-backwards for HMMshape
        gamma, alpha, beta, loglik = hmmFwdBack(pi0, T, lls)  #%given pi0, alpha_nm, and eta_nk (eta_nk from GLM)
        #NOTE: make T depend on X for stimulsu driven state transition!!!??
        
        ### fit GLM for each hidden state %eqs (13.17)
        opts2 = fitopts.copy()
        reg = fitopts.regularize
        qf = reg*sp.sparse.eye(l)
        for kk in range(h):
            opts2.weights = gamma[kk,:]  #update learned GLM weights
            results = GLMfit(y, X, qf, opts2)  
            W[:,kk] = results.w
        
        # print(sum(np.diff(W, axis=1)))
        ### update transition at start and end  %eqs (13.18)
        pi0 = gamma[:,0]
        
        xi = 0  #(%eqs (13.19) & (13.43))
        xi_t = np.zeros(h,h,T_)  #use to store prob of two successive latent state through time
        for tt in range(0,T_-1):
            xi0 = np.outer(alpha[:,tt],(lls[:,tt+1]*beta[:,tt+1])) * T
            xi = xi + xi0/xi0.sum()
            xi_t[:,:,tt] = xi0/xi0.sum()
        
#        xi = np.multiply(xi, 1/gamma.sum(1)[:,None])
#        T = np.multiply(xi, 1/xi.sum(1))
        xi = np.multiply(xi, 1/gamma.sum(1))
        
# %% GLM transitions (testing section 2.4 eqn 33 in the Paninski note)
        ### Might have to write custon code for the soft-max fitting???
        opts3 = fitopts.copy()  #weighting by joint prob of successive state
        for jj in range(h):
            for kk in range(h):
                if jj==kk:
#                    opts3.family = 'poibinomlogitssexp'
#                    opts3.weights = xi_t[jj,kk,:]  #update learned GLM weights            
#                    results = GLMfit(y, X, qf, opts3)  
                    K[:,ii,jj] = np.zeros(l)
                else:
                    opts3.family = 'poissexp'
                    opts3.weights = xi_t[jj,kk,:]  #update learned GLM weights
                    results = GLMfit(y, X, qf, opts3)  
                    K[:,ii,jj] = results.w
        
        K = K/(K.sum(1)[:, np.newaxis, :])  #normalized as transition probability
        
# %%
        #T = np.multiply(xi, 1/xi.sum(1))  #transition probability between hidden states
        
        if loglik < loglik0 + thresh:
            print('Converged in ',ii,'iterations:\n')
            break
        loglik0 = loglik.copy()
        print('Iteration ',ii,'log-likelihood = ',loglik0,'\n')
        
    results.W = W
    results.K = K
    results.T = T
    results.p = pi0
    results.loglik = loglik
    results.gamma = gamma
    
    return results

# %% GLM functions
def GLMfit(y, X, qf, opts):
    """
    function for fitting GLM with output y and input X, qf for quadratic regularization/penalty, and opts for optimization options
    opt:
        -w0: initial weights
        -family: GLM family for nonlinearity
        -familyextra: fitting parameters for certain families
        -baseline: a constant baseline output that adds to GLM results
        -weights: weightings on output for specific fitting need
    """
    
    #w0 = opts.w0.copy()  #can be optimized here~
    w0 = np.zeros((X.shape[1],1))
    #w0 = ws[:,1].copy()
    #w0 = np.random.randn(X.shape[1],1)
    w = irls(y, X, w0, qf, opts.baseline, opts.family, opts.familyextra, opts.algo, opts.Display, opts.weights)
    
    l0 = evaMaxGLMLL(y, opts.family, opts.baseline, opts.familyextra, opts.weights)
    lleach = evalGLMLL(y, X, w, opts.baseline, opts.family, opts.familyextra, opts.weights)
    ll = lleach.sum()
    ll = ll - l0
    
    lp = 0.5*np.matmul(np.matmul(w.T, qf.toarray()), w)  #quadratic penalty
    
    results = DotMap()
    results.w = w
    results.loglikelihood = -ll
    results.logpenalty = -lp
    results.loglikelihoodeach = -lleach
    return results

def evalGLMLL(y, X, w, b, family, familyextra, weights):
    ###linear equation
    #r = (X*w).sum(1) + b
    r = np.matmul(X,w) + b
    method = family
    if method == 'normid':
        v = r.copy()
        gain = 1/familyextra**2
        lleach = 0.5/familyextra**2*(y-v)**2
    elif method == 'binomlogit':
        if familyextra == 1:
            gain = 1
            v = 1/(1+np.exp(-r))
            lleach = np.zeros_like(v)
            ol = np.where(y==1)[0]
            zl = np.where(y==0)[0]
            lleach[ol] = np.array(-np.log(v[ol]+eps))
            lleach[zl] = np.array(-np.log(1-v[zl]+eps))
        else:
            y = y/familyextra
            v = 1/(1+np.exp(-r))
            gain = familyextra.copy()
            lleach = -familyextra*(y*np.log(v+eps) + (1-y)*np.log(1-v+eps))
    elif method == 'poissonexp':
        v = np.exp(r)
        gain = 1
        lleach = -y*r + v
    else:
        print('Unsuporrted family')
        
    lleach = lleach*weights
#    ll = lleach.sum()
    
    return np.squeeze(lleach)

def evaMaxGLMLL(y, family, baseline, familyextra, weights):
    """
    evaluate the largest log-likelihood posibile to start with an initial condition
    """
    method = family
    if method == 'normid':
        ll = 0
    elif method == 'binomlogit':
        y = y/familyextra
        ll = -familyextra* ( np.matmul((weights*y).T,np.log(y+eps)) + np.matmul((weights*(1-y)).T,np.log(1-y+eps) ))
    elif method == 'poissexp':
        ll = -np.matmul((weights*y).T , np.log(y+eps)) + sum(weights*y)
    return ll

def irls(y, X, w, qf, b, family, familyextra, algo, Display, weights):
    """
    helper function for scipy optimization
    """
    res = sp.optimize.minimize( evaLplusp, w, args=(y, X, b, qf, family, familyextra, weights))#
#                                , method='Nelder-Mead', method='Nelder-Mead')# , method='BFGS')
#                                ,
#                                ,bounds = ((0,None),(0,None),(None,None),(None,None)))
    w = res.x
#    glm = GLMCV(distr='probit', tol=1e-5,
#            score_metric="deviance",
#            alpha=0., learning_rate=1e-5, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
#    glm.fit(X, y)
#    w = glm.beta_
    
    return w

def evaLplusp(w, y, X, b, qf, family, familyextra, weights):
    """
    objective function for negative log-likelihood minimization
    """
    f0 = evalGLMLL(y, X, w, b, family, familyextra, weights)
    f0 = f0.sum()
    f = f0 + 0.5* np.matmul(np.matmul(w.T,qf.toarray()),w)
    return float(f)  #, g, H

# %% test single GLM
X = np.random.randn(2000,10)   #stimulus
rg = np.arange(0,10)
ww = np.exp(-(rg-3)**2/2/0.8**2)  #kernel
eta = np.matmul(X,ww)-1;  #emission
p = 1/(1 + np.exp(-eta))  #nonlinearity
y = (p > np.random.rand(len(p)))  #spiking process

w0 = ww+np.random.randn(len(ww))*0.5
qf = 0.001*sp.sparse.eye(len(ww))
test_w = irls(y, X, w0, 0.00*qf, 1, 'binomlogit', 1, 1, 1, 1)

plt.figure()
plt.plot(ww)
plt.plot(test_w,'-')
plt.plot(w0,'--o')
# %% HMM fuinctions
def hmmFwdBack(initDist, transmat, softev):
    """
    input:
        Forward-backward algorithm for HMM
        initDist==pi0: length h hidden states
        transmat==alpha_nm:  h x h matrix
        softev==eta_ny: h x T hidden state by time series
    returns:
        gamma: h x T       ->Bayes P(qt = n | Y)
        alpha (a): h x T   ->forward P(y1...yt, qt = n)
        beta (b): h x T    ->backward P(yt+1...yT | qt = n)
        loglik: scalar     ->sum log(P(Y))
    """
    loglik, alpha = hmmFilter(initDist, transmat, softev)
    beta = hmmBackwards(transmat, softev)
    temp = alpha*beta
    gamma = temp/temp.sum(axis=0)  #normalize by column
    return gamma, alpha, beta, loglik

def hmmFilter(initDist, transmat, softev):
    """
    Forward calcultion for alpha
    """
    K, T = softev.shape
    scale = np.zeros(T)
    AT = transmat.T
    alpha = np.zeros((K,T))
    temp = initDist *softev[:,0]
    scale[0] = temp.sum()
    alpha[:,0] = temp/temp.sum()
    for tt in range(1,T):
        temp = np.matmul(AT, alpha[:,tt-1])*softev[:,tt]
        scale[tt] = temp.sum()
        alpha[:,tt] = temp/temp.sum()
    loglik = np.sum(np.log(scale + eps))
    return loglik, alpha
    
def hmmBackwards(transmat, softev):
    """
    Backwards calculation for beta
    """
    K, T = softev.shape
    beta = np.zeros((K,T))
    beta[:,-1] = np.ones(K)
    for tt in range(T-2,-1,-1):  #from T-1 back to 0, strange in python...
        temp = np.matmul(transmat, beta[:,tt+1]*softev[:,tt+1])
        beta[:,tt] = temp/temp.sum()  #normalize by all
    return beta

# %% testing w/ simulated data
#ns = 2000
#X = np.random.randn(ns,10)   #stimulus
#a = np.mod(np.floor(np.arange(0,ns)/50),2) == 1  #states
#
#rg = np.arange(0,10)
#ws = np.array([ np.exp(-(rg-3)**2/2/0.8**2) , np.exp(-(rg-7)**2/2/1.0**2) ]).T  #filters
#W = np.outer(a, ws[:,0]) + np.outer((1-a),ws[:,1])
#
##W = a'*ws(:,1)' + (1-a)'*ws(:,2)';
#
#eta = (X*W).sum(1)-1;  #emission
#p   = 1/(1 + np.exp(-eta));  #nonlinearity
#y = (p > np.random.rand(len(p)))
#
##plt.plot(y)
#plt.figure()
#plt.plot(ws)
#
# %%
##glmopts = {}
##glmopts['family'] = 'binonlogit'
##glmopts['familyextra'] = 1
##glmopts['thresh'] = 10**-3
#glmopts = DotMap()
#glmopts.family = 'binomlogit'
#glmopts.familyextra = 1
#glmopts.regularize = 10
#glmopts.thresh = 10**-5
#glmopts.niter = 100  #maximum iteration
#glmopts.baseline = 1 
#glmopts.algo = 1 
#glmopts.Display = 1
#T0 = np.random.rand(ws.shape[1],ws.shape[1])
#T0 = T0/T0.sum(0)
#w0 = np.random.randn(ws.shape[0],ws.shape[1])*1 + ws
#glmopts.w0 = w0.copy()
#results = GLMHMM(y, X, w0, T0, glmopts)
#
#plt.figure()
#plt.plot(results.W)

# %%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% generative model of GLM-HMM with driven state transitions
def transition_GLM(X,K):
    """
    Given stimulus and kernels, return a simulated discrete state transition time series
    X: Txk  simulus design matrix with time length T and window size k
    K: hxhxk  kernel with lenght k transitioning between h states
    """
    T, k = X.shape
    h, h_, _ = K.shape
    temp = np.exp(np.einsum('ij,jkl->ikl', X, np.einsum('ijk->kij', K)))#(X @ K.T)  #probability of h states unormalized
    lamda_tran = temp/(temp.sum(1)[:, np.newaxis, :])  #normalized probability a_ijt, tranition from i to j at time t
#    lamda_tran = np.array([temp[it,:,:]/temp[it,:,:].sum(1) for it in range(T) ])
    h_states = np.arange(0,k)  #all possible states
    state = np.zeros(T)  #time series of states
    state[0] = np.random.randint(h)
    for tt in range(1,T):
        alpha_t = lamda_tran[tt,:,:]  #alpa_ij at time tt
        last_state = int(state[tt-1])
        cum_p = np.cumsum(alpha_t[:,last_state])
        trans_id = np.where(cum_p > np.random.rand())[0][0]
        state[tt] = h_states[trans_id]
    
    return state

# %%
T = 2000
pad = 10
rg = np.arange(0,pad)
SN = np.random.randn(T,1)
Stimpad = np.concatenate((SN,np.zeros((pad,1))),axis=0)
S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
X = np.squeeze(Stimpad[S])

#K = np.array([[rg*0, np.exp(-(rg-3)**2/2/0.8**2), np.exp(-(rg-3)**2/2/0.8**2)],\
#               [np.exp(-(rg-3)**2/2/0.8**2), rg*0, np.exp(-(rg-3)**2/2/0.8**2)],\
#               [5*np.exp(-(rg-7)**2/2/1.0**2),np.exp(-(rg-7)**2/2/1.0**2), rg*0]])
K = np.array([[np.ones(len(rg))*0, np.exp(-(rg-3)**2/2/0.8**2)],\
               [.01*np.exp(-(rg-7)**2/2/0.8**2), np.ones(len(rg))*0]])
st = transition_GLM(X,K)
plt.plot(st)
#K_tran = np.array([ np.exp(-(rg-3)**2/2/0.8**2) , np.exp(-(rg-7)**2/2/1.0**2) ])  #filters
#lamda_tran = np.exp(X @ K_tran.T)
#lamda_tran = lamda_tran/(lamda_tran.sum(1)[:,None]+1)  #alpha transition matrix though time, +1 for exp(0) same state
#aa = np.zeros(ns)
#aa[0] = np.random.rand() > 0.5  #compare with random
#for tt in range(1,ns):
#    rand = np.random.rand()
#    if aa[tt-1] == True:
#        aa[tt] = (lamda_tran[tt,0]>rand)
#    else:
#        aa[tt] = (lamda_tran[tt,1]>rand)
#
#plt.plot(aa)

# %%
a = st.copy()

rg = np.arange(0,10)
ws = np.array([ np.exp(-(rg-3)**2/2/0.8**2) , np.exp(-(rg-7)**2/2/1.0**2) ]).T  #filters
W = np.outer(a, ws[:,0]) + np.outer((1-a),ws[:,1])
eta = (X*W).sum(1)-1;  #emission
p   = 1/(1 + np.exp(-eta));  #nonlinearity
y = (p > np.random.rand(len(p)))

plt.plot(y)
plt.figure()
plt.plot(ws)

# %% full GLM-HMM inference
glmopts = DotMap()
glmopts.family = 'binomlogit'
glmopts.familyextra = 1
glmopts.regularize = 10
glmopts.thresh = 10**-5
glmopts.niter = 100  #maximum iteration
glmopts.baseline = 1 
glmopts.algo = 1 
glmopts.Display = 1
K0 = np.random.rand(K.shape[0],K.shape[1],K.shape[2])
w0 = np.random.randn(ws.shape[0],ws.shape[1])*1 + ws
glmopts.w0 = w0.copy()
results = GLMHMM(y, X, w0, K0, glmopts)

plt.figure()
plt.plot(results.W)