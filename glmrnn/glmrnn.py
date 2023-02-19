#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:47:29 2023

@author: kschen
"""
import numpy as np
import scipy as sp

class glmrnn:
    
    def __init__(self, N, T, dt, k, kernel_type='tau', nl_type='log-linear', spk_type="Poisson"):
        
        self.N = N
        self.T = T
        self.dt = dt
        self.tau = k
        self.kernel_type, self.nl_type, self.spk_type = kernel_type, nl_type, spk_type
        if nl_type=='sigmoid':    
            self.lamb_max = 1
            self.lamb_min = 0
        if spk_type=="neg-bin":
            self.rnb = 0.1
        self.W = np.random.randn(N,N)*0.1  # initial network
        self.b = np.random.randn(N)*0.1  # initial baseline
        self.U = np.random.randn(N)*0.1  # initial input vector
        self.data = [] 
        
    def forward(self, ipt):
        """
        forward simulation of GLM-RNN given parameters
        """
        spk = np.zeros((self.N, self.T))
        rt = np.zeros((self.N, self.T))
        for tt in range(self.T-1):
            lamb = self.W @ rt[:,tt] + self.b + self.U*ipt[tt]
            spk[:,tt+1] = self.spiking(self.nonlinearity(lamb*self.dt))
            rt[:,tt+1] = self.kernel(rt[:,tt] , spk[:,tt])
        self.data = (spk,rt,ipt)
        return spk, rt
    
    def nonlinearity(self, x):
        """
        Nonlinearity for spiking function
        I: input rate vector N:
        O: nonlinear spiking potential vector N:
        """
        if self.nl_type=='exp':
            nl = np.exp(x)
        if self.nl_type=='log-linear':
            nl = np.log((1+np.exp(x)))
        if self.nl_type=='sigmoid':
            nl = (self.lamb_max-self.lamb_min)/(1+np.exp(-x)) + self.lamb_min    
        return nl
    
    def spiking(self, nl):
        """
        Spiking process that emits descrete spikes given continuous rate
        I: rate vector N:
        O: spike vector N:
        """
        if self.spk_type=='Poisson':
            spk = np.random.poisson(nl)
        if self.spk_type=='Bernoulli':
            spk = np.random.binomial(1,nl)
        if self.spk_type=='neg-bin':
            spk = np.random.negative_binomial(self.rnb+nl, self.rnb)
        return spk
    
    def kernel(self, rt, spk):
        """
        Linear kernel operation on spikes
        I: spike trains N x T, and past rate N x T
        O: convolved spikes N x T
        """
        if self.kernel_type=='tau':
            filt = (1-self.dt/self.tau)*rt + self.dt/self.tau*spk
        return filt
    
    #@classmethod
    def kernel_filt(self, spk):
        """
        
        """
        rt = np.zeros((self.N, self.T))
        for tt in range(self.T-1):
            rt[:,tt+1] = self.kernel(rt[:,tt] , spk[:,tt])
        return rt
        
        
    def neg_log_likelihood(self, ww, spk, rt, ipt=None, lamb=0):
        """
        Negative log-likelihood calculation
        I: parameter vector, spike, rate, input, and regularization
        O: negative log-likelihood
        """
        b,U,W = self.vec2param(ww)
        ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)) \
                - self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
                - lamb*np.linalg.norm(W)
        return -ll
    
    def param2vec(self, b,U,W):
        ww = np.concatenate((b.reshape(-1),U.reshape(-1),W.reshape(-1)))
        return ww
    
    def vec2param(self, ww):
        b = ww[:self.N]
        U = ww[self.N:2*self.N]
        W = ww[2*self.N:].reshape(self.N,self.N)
        return b,U,W
    
    def fitting(self, lamb=0):
        """
        MLE fit for GLM-RNN parameters
        I: regularization parameter
        O: fitting summary
        """
        spk,rt,ipt = self.data
        dd = self.N**2 + self.N + self.N
        w_init = np.ones([dd,])*0.1
        res = sp.optimize.minimize(lambda w: self.neg_log_likelihood(w, \
                        spk, rt, ipt, lamb), w_init, method='L-BFGS-B')
        w_map = res.x
        self.b, self.U, self.W = self.vec2param(w_map)
        
        return res.fun, res.success
    