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

global eps
eps = 10**-15  #flow-point precision
    
# %% main function
def GLMHMM(y, X, W0, T0, fitopts):
    """
    takes time series of observation y, stimulus X, initial GLM weights W0 and state-transitions T0
    for fitting with firopts for the model (GLM type, iterations, threshold etc.)
    returns result class with tranistion matrix and GLM results
    y: length time T
    X: length time T x w stimulus window (the design matrix)
    W0: l x h (length of kernel x hidden states)
    T0: h x h
    """
    ###import dimensions and initialize probabilities
    thresh = fitopts.thresh #10**-3
    T_ = len(y)
    h = W0.shape[1]
    l = X.shape[1]  #might change due to basis function merthod for kernel !!??
    pi0 = np.ones(h)
    pi0 = pi0/sum(pi0)
    
    W = W0.copy()
    T = T0.copy()
    niter = fitopts.niter
    lls = np.zeros((h,T_))  #hidden state x time series
    loglik0 = -np.infty

    for ii in range(0,niter):   #iteration loop
        etas = np.matmul(X,W)
        for kk in range(h):   #hidden state loop
            lls[kk,:] = evalGLMLL( y, etas[:,kk][:,None], np.ones(1)[:,None], 0, fitopts.family, fitopts.familyextra, 1)
        lls = np.exp(-lls)
            
        ### Forward-backwards for HMMshape
        gamma, alpha, beta, loglik = hmmFwdBack(pi0, T, lls)  #%given pi0, alpha_nm, and eta_nk (eta_nk from GLM)
        #NOTE: make T depend on X for stimulsu driven state transition!!!??
        
        ### fit GLM for each hidden state %eqs (13.17)
        opts2 = fitopts.copy()
        for kk in range(h):
            opts2.weights = gamma[kk,:]  #update learned GLM weights
            results = GLMfit(y, X, 0.001*sp.sparse.eye(l), opts2)  
            W[:,kk] = results.w
        
        ### update transition at start and end  %eqs (13.18)
        pi0 = gamma[:,0]
        
        xi = 0  #(%eqs (13.19) & (13.43))
        for tt in range(0,T_-1):
            xi0 = np.outer(alpha[:,tt],(lls[:,tt+1]*beta[:,tt+1])) * T
            xi = xi + xi0/xi0.sum()
        
#        xi = np.multiply(xi, 1/gamma.sum(1)[:,None])
#        T = np.multiply(xi, 1/xi.sum(1))
        xi = np.multiply(xi, 1/gamma.sum(1))
        T = np.multiply(xi, 1/xi.sum(1))
        
#        if loglik < loglik0 + thresh:
#            print('Converged in ',ii,'iterations:\n')
#            break
        loglik0 = loglik.copy()
        print('Iteration ',ii,'log-likelihood = ',loglik0,'\n')
        
    results.W = W
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
    #w0 = np.zeros((X.shape[1],1))
    #w0 = ws[:,1].copy()
    w0 = np.random.randn(X.shape[1],1)
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
                                #, method='Nelder-Mead', method='Nelder-Mead')# , method='BFGS')
                                #,
                                #,bounds = ((0,None),(0,None),(None,None),(None,None)))
    w = res.x
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
test_w = irls(y, X, w0, 0.00*qf, 1, glmopts.family, glmopts.familyextra, 1, 1, 1)

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
    K, T = softev.shape
    scale = np.zeros(T)
    AT = transmat.T
    alpha = np.zeros((K,T))
    temp = initDist[:]*softev[:,0]
    alpha[:,0], scale[0] = temp/temp.sum()
    for tt in range(1,T):
        temp = np.matmul(AT, alpha[:,tt-1])*softev[:,tt]
        alpha[:,tt], scale[tt] = temp/temp.sum()
    loglik = np.sum(np.log(scale + eps))
    return loglik, alpha

def hmmBackwards(transmat, softev):
    K, T = softev.shape
    beta = np.zeros((K,T))
    beta[:,-1] = np.ones(K)
    for tt in range(T-2,-1,-1):
        temp = np.matmul(transmat, beta[:,tt+1]*softev[:,tt+1])
        beta[:,tt] = temp/temp.sum()  #normal(ize by all
    return beta

# %% testing w/ simulated data
ns = 2000
X = np.random.randn(ns,10)   #stimulus
a = np.mod(np.floor(np.arange(0,ns)/50),2) == 1  #states

rg = np.arange(0,10)
ws = np.array([ np.exp(-(rg-3)**2/2/0.8**2) , np.exp(-(rg-7)**2/2/1.0**2) ]).T  #filters
W = np.outer(a, ws[:,0]) + np.outer((1-a),ws[:,1])

#W = a'*ws(:,1)' + (1-a)'*ws(:,2)';

eta = (X*W).sum(1)-1;  #emission
p   = 1/(1 + np.exp(-eta));  #nonlinearity
y = (p > np.random.rand(len(p)))

#plt.plot(y)
plt.figure()
plt.plot(ws)

# %%
#glmopts = {}
#glmopts['family'] = 'binonlogit'
#glmopts['familyextra'] = 1
#glmopts['thresh'] = 10**-3
glmopts = DotMap()
glmopts.family = 'binomlogit'
glmopts.familyextra = 1
glmopts.thresh = 10**-5
glmopts.niter = 100
glmopts.baseline = 1 
glmopts.algo = 1 
glmopts.Display = 1
T0 = np.random.rand(ws.shape[1],ws.shape[1])
T0 = T0/T0.sum(0)
w0 = np.random.randn(ws.shape[0],ws.shape[1])
glmopts.w0 = w0.copy()
results = GLMHMM(y, X, w0, T0, glmopts)

plt.figure()
plt.plot(results.W)