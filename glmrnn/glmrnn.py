#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:47:29 2023

@author: kschen
"""
import numpy as np
import scipy as sp
from ssm.regression import fit_scalar_glm

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
#        self.data = (spk,rt,ipt)
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
        given spike train produce synaptic filtered rate
        I: spk: N x T
        O: rt: N x T
        """
        rt = np.zeros((self.N, self.T))
        for tt in range(self.T-1):
            print(tt)
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
    
    def fit_single(self, data, lamb=0):
        """
        MLE fit for GLM-RNN parameters
        I: regularization parameter
        O: fitting summary
        """
        spk,ipt = data
        rt = self.kernel_filt(spk)
        dd = self.N**2 + self.N + self.N
        w_init = np.ones([dd,])*0.1
        res = sp.optimize.minimize(lambda w: self.neg_log_likelihood(w, \
                        spk, rt, ipt, lamb), w_init, method='L-BFGS-B')
        w_map = res.x
        self.b, self.U, self.W = self.vec2param(w_map)
        
        return res.fun, res.success
    
    def fit_batch(self, data):
        """
        MLE fit for GLM-RNN parameters, with batch and with ssm function
        I: data: (list(spk), list(rt), list(ut))
                spk: TxN  (from ssm by default)
                ut: T x input_dim
        O: None... fitting parameters in class
        """
        l_spk, l_ut = data
        reps = len(l_spk)
        Xs = []
        ys = []
        param_W = np.zeros((self.N, self.N))
        param_b = np.zeros(self.N)
        param_U = np.zeros(self.N)
        for n in range(self.N):  # loop through neurons, given that we assume indpendence
            for r in range(reps):
                rt_r = self.kernel_filt(l_spk[r].T).T # T x N rate matrix
                tempx = np.concatenate((rt_r, l_ut[r]),1)  # design matrix is rate and input
                Xs.append(tempx)
                ys.append(l_spk[r][:,n])
            thetas = self.fit_glm_local(Xs,ys)  # fitting per neuron observation
            param_W[n,:] = thetas[0][:-1]  # neuron weights
            param_U[n] = thetas[0][-1]  # input weights
            param_b[n] = thetas[1]  # baseline rate
        
        self.W = param_W
        self.U = param_U
        self.b = param_b
        return
    
    def fit_glm_local(self, Xs, ys):
        """
        helper function to hide ssm glm-fitting function
        """
        theta = fit_scalar_glm(Xs, ys,
                       model="poisson",
                       mean_function="softplus",
                       model_hypers={},
                       fit_intercept=True,
                       weights=None,
                       X_variances=None,
                       prior=None,
                       proximal_point=None,
                       threshold=1e-6,
                       step_size=1,
                       max_iter=50,
                       verbose=False)
        return theta
    #####
    # IDEA: with 'weight' parameter normaly use in ssm training, develop an algorithm for GLM-RNN,
    #       that as similar weighting for state learning...
    #####
    