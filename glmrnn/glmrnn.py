#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:47:29 2023

@author: kschen
"""
#import numpy as np
import autograd.numpy as np
import scipy as sp
from autograd.scipy.special import logsumexp
from ssm.regression import fit_scalar_glm

from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd

eps = 10**-10
class glmrnn:
    
    def __init__(self, N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson"):
        
        self.N = N
        self.T = T
        self.dt = dt
        self.tau = tau
        self.kernel_type, self.nl_type, self.spk_type = kernel_type, nl_type, spk_type
        if nl_type=='sigmoid':    
            self.lamb_max = 1
            self.lamb_min = 0
        if spk_type=="neg-bin":
            self.rnb = 0.1
        if spk_type=='Gaussian':
            self.std = 0.1
        self.W = np.random.randn(N,N)*0.1  # initial network
        self.b = np.random.randn(N)*0.1  # initial baseline
        self.U = np.random.randn(N)*0.1  # initial input vector
        self.data = [] 
        self.K = 2 # number of states for state-transitions
        if kernel_type=='basis':
            self.nbasis = 4  # number of basis functions
            nbins = int(np.floor(tau/dt))
            self.nbins = nbins  # time window tiled
            self.W = np.random.randn(N,N,4)*0.1  # tensor with kernel weights
        
    def forward(self, ipt=None):
        """
        forward simulation of GLM-RNN given parameters
        """
        if self.kernel_type != 'basis':
            if ipt is None:
                ipt = np.zeros(self.T)
            spk = np.zeros((self.N, self.T))
            rt = np.zeros((self.N, self.T))
            for tt in range(self.T-1):
                lamb = self.W @ rt[:,tt] + self.b + self.U*ipt[tt]
                spk[:,tt] = self.spiking(self.nonlinearity(lamb)*self.dt)
                rt[:,tt+1] = self.kernel(rt[:,tt] , spk[:,tt])
    #        self.data = (spk,rt,ipt)
            if self.spk_type=='Gaussian':
                return spk, rt
        
        if self.kernel_type == 'basis':
            if ipt is None:
                ipt = np.zeros(self.T+self.nbins)  # padding for kernel window
            else:
                ipt = np.concatenate((ipt.squeeze(), np.zeros(self.nbins)))
            spk = np.zeros((self.N, self.T+self.nbins))
            rt = np.zeros((self.N, self.T+self.nbins))
            Wijk = self.weight2kernel()
            for tt in range(self.nbins,self.T):
                ut = self.b + self.U*ipt[tt] + \
                     np.einsum('ijk,jk->i',  Wijk, spk[:,tt-self.nbins:tt])  #neat way for linear dynamics
                ut = self.nonlinearity(ut)*self.dt
                rt[:,tt] = ut
                spk[:,tt] = self.spiking(ut)
            spk = spk[:,self.nbins:]  # remove padding
            rt = rt[:,self.nbins:]
        return spk.astype(int), rt
    
    def nonlinearity(self, x):
        """
        Nonlinearity for spiking function
        I: input rate vector N:
        O: nonlinear spiking potential vector N:
        """
        if self.nl_type=='exp':
            nl = np.exp(x)
        if self.nl_type=='log-linear':
            nl = np.log((1+np.exp(x))+eps)
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
        if self.spk_type=='Gaussian':
            spk = nl*1 + np.random.randn()*self.std
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
    def kernel_filt(self, spk, Wijk=None):
        """
        given spike train produce synaptic filtered rate
        I: spk: N x T
        O: rt: N x T
        """
        if self.kernel_type != 'basis':
            rt = np.zeros((self.N, self.T))
            for tt in range(self.T-1):
                rt[:,tt+1] = self.kernel(rt[:,tt] , spk[:,tt])
        elif self.kernel_type == 'basis':
            rt = np.zeros((self.N, self.T+self.nbins))
            if Wijk is None:
                Wijk = self.weight2kernel()
            for tt in range(self.nbins, self.T):
                rt[:,tt] = np.einsum('ijk,jk->i',  Wijk, spk[:,tt-self.nbins:tt])
            rt = rt[:,self.nbins:]
        return rt
    
    def weight2kernel(self, weights=None):
        """
        Turning W into tensor of coupling kernels Wijk
        """
        Wijk = np.zeros((self.N, self.N, self.nbins))
        if weights is None:
            for ii in range(self.N):
                for jj in range(self.N):
                    Wijk[ii,jj,:] = self.W[ii,jj,:] @ self.basis_function().T
        else:
            for ii in range(self.N):
                for jj in range(self.N):
                    Wijk[ii,jj,:] = weights[ii,jj,:] @ self.basis_function().T
        return Wijk
    
    def basis_function(self):
        """
        Raised cosine basis function to tile the time course of the response kernel
        nkbins of time points in the kernel and nBases for the number of basis functions
        """
        nkbins = self.nbins
        nBases = self.nbasis
        ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
        dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
        width = 5.*dbcenter # width of each bump
        bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
        def bfun(x,period):
            return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
        temp = ttb - np.tile(bcenters,(nkbins,1)).T
        BBstm = np.array([bfun(xx,width) for xx in temp]).T
        return BBstm  # output bin x basis
        
    def neg_log_likelihood(self, ww, spk, rt, ipt=None, state=None, lamb=0):
        """
        Negative log-likelihood calculation
        I: parameter vector, spike, rate, input, and regularization
        O: negative log-likelihood
        """
        if state is None:
            b,U,W = self.vec2param(ww)
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)+eps) \
                    - self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
                    - lamb*np.linalg.norm(W)
            return -ll
        elif state is not None:
            b,U,W,ws = self.vec2param(ww, True)
#            ws = ww[-(self.K*self.N):].reshape(self.K, self.N)  #state readout weights
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)+eps) \
                    - self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
                    - lamb*np.linalg.norm(W)
            lp_states = ws @ rt #np.exp(ws @ rt_true) #
            # lp_states = lp_states / lp_states.sum(0)[None,:]  # P of class probablity
            lp_states = lp_states - logsumexp(lp_states,0)[None,:]  # logP
            onehot = self.state2onehot(state)  # one-hot of true states
            state_cost = -np.sum(onehot * lp_states)        
            return -ll + state_cost
        
    def neg_log_likelihood_wo_input(self, ww, spk, rt, state=None, lamb=0):
        """
        Negative log-likelihood calculation without input vector
        I: parameter vector, spike, rate, input, and regularization
        O: negative log-likelihood
        """
        if state is None:
            b,W = self.vec2param(ww,ipt=False)
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None]) + eps) \
                    - self.nonlinearity(W @ rt + b[:,None])*self.dt) \
                    - lamb*np.linalg.norm(W)
            return -ll        
        elif state is not None:
            b,U,W,ws = self.vec2param(ww, state=True,ipt=True)
#            ws = ww[-(self.K*self.N):].reshape(self.K, self.N)  #state readout weights
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None])+eps) \
                    - self.nonlinearity(W @ rt + b[:,None])*self.dt) \
                    - lamb*np.linalg.norm(W)
            lp_states = ws @ rt #np.exp(ws @ rt_true) #
            # lp_states = lp_states / lp_states.sum(0)[None,:]  # P of class probablity
            lp_states = lp_states - logsumexp(lp_states,0)[None,:]  # logP
            onehot = self.state2onehot(state)  # one-hot of true states
            state_cost = -np.sum(onehot * lp_states)  
            return -ll + state_cost
        
    def neg_log_likelihood_kernel(self, ww, spk, rt=None, ipt=None, lamb=0):
        """
        Negative log-likelihood function for interacting kernel functions
        I: parameter vector, spike train, and regularization
        O: negative log-likelihood
        """
        # unpack parameters
        b = ww[:self.N]
        U = ww[self.N:self.N*2]
        W = ww[self.N*2:].reshape(self.N, self.N, self.nbasis)
        Wijk = self.weight2kernel(W)
        lamb = self.kernel_filt(spk, Wijk) + b[:,None] + U[:,None]*ipt.T
        print(lamb.shape)
        ll = (spk * np.log(self.nonlinearity(lamb)+eps) \
                    - self.nonlinearity(lamb)*self.dt) \
                    - lamb*np.linalg.norm(W)
        ll = np.sum(ll)
        return -ll
           
    def state2onehot(self, states, K=None):
        """
        state vector to onehot encoding, used for state-constrained likelihood
        """
        if K is None:
            nstate = np.max(states) + 1
        else:
            nstate = K
        T = len(states)
        onehot = np.zeros((nstate,T))
        for tt in range(T):
            onehot[int(states[tt]),tt] = 1
        return onehot

    def param2vec(self, b,U,W):
        ww = np.concatenate((b.reshape(-1),U.reshape(-1),W.reshape(-1)))
        return ww
    
    def vec2param(self, ww, state=False, ipt=True):
        ### baseline vector
        b = ww[:self.N]
        ### with input vector
        if ipt is True:
            U = ww[self.N:2*self.N]
            ### without state readout
            if state is False:
                W = ww[2*self.N:].reshape(self.N,self.N)
                return b, U, W
            ### with state readout
            if state is True:
                W = ww[2*self.N:2*self.N+self.N*self.N].reshape(self.N,self.N)
                ws = ww[2*self.N+self.N*self.N:].reshape(self.K,self.N)  # state readout
                return b, U, W, ws
        ### without input vector
        elif ipt is False:
            if state is False:
                W = ww[self.N:].reshape(self.N,self.N)
                return b, W
            if state is True:
                W = ww[self.N:self.N+self.N*self.N].reshape(self.N,self.N)
                ws = ww[self.N+self.N*self.N:].reshape(self.K,self.N)  # state readout
                return b, U, W, ws
    
    def fit_single(self, data, lamb=0):
        """
        MLE fit for GLM-RNN parameters
        I: regularization parameter
        O: fitting summary
        """
        spk,ipt = data
        rt = self.kernel_filt(spk)
        if ipt is None:
            dd = self.N**2 + self.N
            w_init = np.ones([dd,])*0.
            res = sp.optimize.minimize(lambda w: self.neg_log_likelihood_wo_input(w, \
                        spk, rt, None, lamb), w_init, method='L-BFGS-B')
            w_map = res.x
            self.b, self.W = self.vec2param(w_map, ipt=False)
            self.U *= 0
        elif ipt is not None: 
            dd = self.N**2 + self.N + self.N
            w_init = np.ones([dd,])*0.
            res = sp.optimize.minimize(lambda w: self.neg_log_likelihood(w, \
                        spk, rt, ipt, None, lamb), w_init, method='L-BFGS-B')
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
                tempx = np.concatenate((rt_r, l_ut[r]),1)*self.dt  # design matrix is rate and input
                Xs.append(tempx)
                ys.append(l_spk[r][:,n])
#            thetas = self.fit_glm_local(Xs,ys)  # fitting per neuron observation
            thetas = self.fit_glm(Xs, ys)
            param_W[n,:] = thetas[0][:-1]  # neuron weights
            param_U[n] = thetas[0][-1]  # input weights
            param_b[n] = thetas[1]  # baseline rate
        
        self.W = param_W
        self.U = param_U
        self.b = param_b
        return
    
    def fit_glm_local(self, Xs, ys):
        """
        helper function to hide ssm glm-fitting function... not sure if it works
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
    
    def log_marginal(self, ww, spks, ipts, state=None):
        """
        summing over list for log-marginal calculation
        """
        ll = 0
        if ipts[0] is None:
            if state is None:
                for spk,ipt in zip(spks, ipts):
                    rt = self.kernel_filt(spk.T)
                    lli = -self.neg_log_likelihood_wo_input(ww, spk.T, rt)
                    ll += np.sum(lli)
            else:
                for spk,ipt,stt in zip(spks, ipts, state):
                    rt = self.kernel_filt(spk.T)
                    lli = -self.neg_log_likelihood_wo_input(ww, spk.T, rt, stt)
                    ll += np.sum(lli)
            return ll
        elif ipts[0] is not None:
            if state is None:
                for spk,ipt in zip(spks, ipts):
                    rt = self.kernel_filt(spk.T)
                    lli = -self.neg_log_likelihood(ww, spk.T, rt, ipt)
                    ll += np.sum(lli)
            else:
                for spk,ipt,stt in zip(spks, ipts, state):
                    rt = self.kernel_filt(spk.T)
                    lli = -self.neg_log_likelihood(ww, spk.T, rt, ipt, stt)
                    ll += np.sum(lli)
            return ll
    
    def fit_batch_sp(self, data, lamb=0):
        """
        MLE fit for GLM-RNN parameters
        I: data with lists for spikes and inputs
        O: fitting summary
        """
        l_spk, l_ut = data
        dd = self.N**2 + self.N + self.N
        params = np.ones([dd,])*0.1
        def _objective(params):
            obj = self.log_marginal(params, l_spk, l_ut)
            return -obj
        
        res = sp.optimize.minimize(lambda w: _objective(w), params, method='L-BFGS-B')
        w_map = res.x
        self.b, self.U, self.W = self.vec2param(w_map)
        
        return res.fun, res.success
    
    # @ensure_args_are_lists
    def fit_glm(self, data, num_iters=1000, optimizer="bfgs", **kwargs):
        """
        Borrowing the ssm package method
        """
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop,
                         sgd=sgd)[optimizer]
        l_spk, l_ut = data
        if l_ut[0] is None:
            params = np.ones(self.N**2 + self.N)
        elif l_ut[0] is not None:
            params = np.ones(self.N**2 + self.N*2)
        
        def _objective(params, itr):
            obj = self.log_marginal(params, l_spk, l_ut)
            return -obj

        params = optimizer(_objective,
                                params,
                                num_iters=num_iters,
                                **kwargs)
        if l_ut[0] is None:
            self.b, self.W = self.vec2param(params, state=False, ipt=False)
            self.U *= 0
        elif l_ut[0] is not None:
            self.b, self.U, self.W = self.vec2param(params)
        
    #####
    # IDEA: with 'weight' parameter normaly use in ssm training, develop an algorithm for GLM-RNN,
    #       that as similar weighting for state learning...
    #####
    
    def fit_glm_states(self, data, k_states, num_iters=1000, optimizer="bfgs", **kwargs):
        """
        fitting GLM network with state constraints
        """
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop,
                         sgd=sgd)[optimizer]
        l_spk, l_ut, l_state = data
        self.K = k_states#np.max(l_state[0]) + 1  # number of states
        if l_ut[0] is None:
            params = np.ones(self.N**2 + self.N + self.N*self.K)
        elif l_ut[0] is not None:
            params = np.ones(self.N**2 + self.N*2 + self.N*self.K)
        def _objective(params, itr):
            obj = self.log_marginal(params, l_spk, l_ut, l_state)
            return -obj

        params = optimizer(_objective,
                                params,
                                num_iters=num_iters,
                                **kwargs)
        if l_ut[0] is None:
            self.b, self.W, self.ws = self.vec2param(params,state=True, ipt=False)
            self.U *= 0
        elif l_ut[0] is not None:
            self.b, self.U, self.W, self.ws = self.vec2param(params,state=True, ipt=True)
#        self.ws = params[(self.N**2 + self.N*2):].reshape(self.K,self.N)  # state x neurons
        
        