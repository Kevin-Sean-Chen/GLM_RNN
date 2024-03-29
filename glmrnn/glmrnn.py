#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:47:29 2023

@author: kschen
"""
#import numpy as np
import autograd.numpy as np
import scipy as sp
from scipy.linalg import hankel
from autograd.scipy.special import logsumexp
from ssm.regression import fit_scalar_glm

from ssm.util import ensure_args_are_lists
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs

eps = 10**-10
class glmrnn:
    
    def __init__(self, N, T, dt, tau, kernel_type='tau', nl_type='log-linear', spk_type="Poisson"):
        
        self.N = N
        self.T = T
        self.dt = dt
        self.tau = tau
        self.kernel_type, self.nl_type, self.spk_type = kernel_type, nl_type, spk_type
        if nl_type=='sigmoid':    
            self.lamb_max = 10
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
        self.nbins = 1  # default
        if kernel_type=='basis':
            nbasis = 2
            self.nbasis = nbasis  # number of basis functions
            nbins = int(np.floor(tau/dt))
            self.nbins = nbins  # time window tiled
            self.W = np.random.randn(N,N,nbasis)*0.1  # tensor with kernel weights
        self.lamb = 0  # regularization
        self.lamb2 = 0  # for testing other regularization terms
        
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
                subthreshold = self.W @ rt[:,tt] + self.b + self.U*ipt[tt]
                spk[:,tt] = self.spiking(self.nonlinearity(subthreshold)*self.dt)
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
    
    def forward_rate(self, ipt=None):
        if ipt is None:
            ipt = np.zeros(self.T)
        xt = np.zeros((self.N, self.T))
        rt = np.zeros((self.N, self.T))
        for tt in range(self.T-1):
            xt[:,tt+1] = (1-self.dt/self.tau)*xt[:,tt] + self.dt*( \
                          self.W @ rt[:,tt] + self.b + self.U*ipt[tt]) \
                          + np.random.randn(self.N)*np.sqrt(self.dt)*self.noise
            rt[:,tt+1] = self.nonlinearity(xt[:,tt+1]*self.dt)
        return xt, rt
    
    def nonlinearity(self, x):
        """
        Nonlinearity for spiking function
        I: input rate vector N:
        O: nonlinear spiking potential vector N:
        """
        if self.nl_type=='exp':
            nl = np.exp(x)
        if self.nl_type=='log-linear':
#            nl = np.log((1+np.exp(x))+eps)
            nl = self.stable_softmax(x)
        if self.nl_type=='sigmoid':
            nl = (self.lamb_max-self.lamb_min)/(1+np.exp(-x)) + self.lamb_min    
        return nl
    
    def stable_softmax(self, x, log=None):
        """
        Stable softmax and log-likelihood
        """
        low_cut = -20
        high_cut = 500
        
        if log is None:
            f = np.log(1+np.exp(x))
            if len(np.where(x < low_cut)[0]) > 0:
                iix = np.where(x < low_cut)[0]
                f[iix] = np.exp(x[iix])
            if len(np.where(x > high_cut)[0]) > 0:
                iix = np.where(x > high_cut)[0]
                f[iix] = x[iix]
        if log is True:
            f = np.log(np.log(1+np.exp(x)))
            if len(np.where(x < low_cut)[0]) > 0:
                iix = np.where(x < low_cut)[0]
                f[iix] = x[iix]
            if len(np.where(x > high_cut)[0]) > 0:
                iix = np.where(x > high_cut)[0]
                f[iix] = np.log(x[iix])
        return f
    
    def sigmoid_ll(self, y_true, y_pred):
        """
        Compute log-likelihood of sigmoid function
        """
        ll = np.sum( y_true*np.log(y_pred) + (self.lamb_max - y_true)*np.log(self.lamb_max - y_pred))
        return ll

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
#            rt = np.zeros((self.N, self.T+self.nbins))
#            rt = []
#            if Wijk is None:
#                Wijk = self.weight2kernel()
#            spk_ = np.concatenate((np.zeros((self.N,self.nbins)),spk),1)  # padding
#            for tt in range(self.nbins, self.T+self.nbins):
##                rt[:,tt] = np.einsum('ijk,jk->i',  Wijk, spk[:,tt-self.nbins:tt])
#                rt.append(np.einsum('ijk,jk->i',  Wijk, spk_[:,tt-self.nbins:tt]))
#            rt = np.array(rt).T
            ### too slow... try matrix method~~
            # rt = self.design_matrix(spk)
            rt = self.design_matrix_proj(spk, Wijk)
            ###
#            rt = rt[:,self.nbins:]
        return rt
    
    def design_matrix_proj(self, spk, weights):
        """
        return matrix that has spikes projected onto weighted kernels
        spk:      TxN spiking pattern
        weights:  N*nbasis weight vector
        Output:   X is (N*nbasis) x T
        """  
        N, T = spk.shape  # time and number of neurons
        D = self.nbins+1  # length of padding for kernels
#        k_weights = weights.reshape(self.N, self.N, self.nbasis)  # reshape to N x nbasis
        Wijk = self.weight2kernel(weights.reshape(self.N,self.N,self.nbasis), flip=True)  # N x N x nbins
        y = spk*1  #spiking patterns
#        basis = self.basis_function()  # D x nbasis
        rt = []  # the projected firing rate
        for ni in range(N):  ### fix this###
            yi = y[ni,:]  # ith neuron
            Xi = sp.linalg.hankel(np.append(np.zeros(D-2),yi[:T-D+2]),yi[T-D+1:])  # design matrix
            rj = []
            for nj in range(N):
#                rt[nj,:] = rt[nj,:] + Xi @ Wijk[ni,nj,:]  # adding the projection onto rate of that neuron
                rj.append(Xi @ Wijk[ni,nj,:])
            rt.append(rj)
        return np.array(rt).sum(0)
    
          
    def weight2kernel(self, weights=None, flip=False):
        """
        Turning W into tensor of coupling kernels Wijk
        """
#        Wijk = np.zeros((self.N, self.N, self.nbins))
        Wijk = []
        if flip is True:
            basis = np.flipud(self.basis_function())
        else:
            basis = self.basis_function()
        if weights is None:
            for ii in range(self.N):
                for jj in range(self.N):
#                    Wijk[ii,jj,:] = self.W[ii,jj,:] @ self.basis_function().T
                    temp = self.W[ii,jj,:] @ basis.T
                    Wijk.append(temp)
            Wijk = np.array(Wijk).reshape(self.N, self.N, self.nbins)
        elif weights is not None:
            for ii in range(self.N):
                for jj in range(self.N):
                    temp = weights[ii,jj,:] @ basis.T
                    Wijk.append(temp)
            Wijk = np.array(Wijk).reshape(self.N, self.N, self.nbins)
                    ### debugging
#                    try:
#                        Wijk[ii,jj,:] = weights[ii,jj,:] @ self.basis_function().T
#                    except ValueError as e:
#                        print("Error: ", e)
#                        print(Wijk[ii,jj,:])
#                        print((weights[ii,jj,:] @ self.basis_function().T))
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
        BBstm = np.flipud(np.array([bfun(xx,width) for xx in temp]).T)
        return BBstm  # output bin x basis
        
    def neg_log_likelihood(self, ww, spk, rt, ipt=None, state=None, lamb=0):
        """
        Negative log-likelihood calculation
        I: parameter vector, spike, rate, input, and regularization
        O: negative log-likelihood
        """
        if state is None:
            b,U,W = self.vec2param(ww)
            ### testing sigmoid
#            ll = self.sigmoid_ll(spk, self.nonlinearity((W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt)) \
#                    - self.lamb*np.linalg.norm(W) \
#                    - self.lamb2*np.sum(np.linalg.norm(W, axis=1))
                    
            ### direct log
#            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)+eps) \
#                    - self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
#                    - self.lamb*np.linalg.norm(W) \
#                    - self.lamb2*np.sum(np.linalg.norm(W, axis=1))
            
            ### stable peice-wise log for softmax
#            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)+eps) \
            ll = np.sum(spk * self.stable_softmax(W @ rt + b[:,None] + U[:,None]*ipt.T , log=True) \
                    - self.stable_softmax(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
                    - self.lamb*np.linalg.norm(W) \
                    - self.lamb2*np.sum(np.linalg.norm(W, axis=1))
#                    - self.lamb2*np.linalg.norm((np.eye(self.N)-W@W.T)) \
#                    - self.lamb*(np.linalg.norm(W) + np.linalg.norm(U))
#                    
            return -ll
        elif state is not None:
            b,U,W,ws = self.vec2param(ww, True)
#            ws = ww[-(self.K*self.N):].reshape(self.K, self.N)  #state readout weights
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)+eps) \
                    - self.nonlinearity(W @ rt + b[:,None] + U[:,None]*ipt.T)*self.dt) \
                    - self.lamb*np.linalg.norm(W)
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
                    - self.lamb*np.linalg.norm(W)
            return -ll        
        elif state is not None:
            b,U,W,ws = self.vec2param(ww, state=True,ipt=True)
#            ws = ww[-(self.K*self.N):].reshape(self.K, self.N)  #state readout weights
            ll = np.sum(spk * np.log(self.nonlinearity(W @ rt + b[:,None])+eps) \
                    - self.nonlinearity(W @ rt + b[:,None])*self.dt) \
                    - self.lamb*np.linalg.norm(W)
            lp_states = ws @ rt #np.exp(ws @ rt_true) #
            # lp_states = lp_states / lp_states.sum(0)[None,:]  # P of class probablity
            lp_states = lp_states - logsumexp(lp_states,0)[None,:]  # logP
            onehot = self.state2onehot(state)  # one-hot of true states
            state_cost = -np.sum(onehot * lp_states)  
            return -ll + state_cost
        
    def neg_log_likelihood_kernel(self, ww, spk, rt=None, ipt=None, lamb=10):
        """
        Negative log-likelihood function for interacting kernel functions
        I: parameter vector, spike train, and regularization
        O: negative log-likelihood
        """
        # unpack parameters
#        b, U, Wijk = self.vec2param_kernel(ww, full_kernel=True)
#        _,_,W = self.vec2param_kernel(ww, full_kernel=False)
#        subthreshold = self.kernel_filt(spk, Wijk) + b[:,None] + U[:,None]*ipt.T
        b = ww[:self.N]
        U = ww[self.N:self.N*2]
        W = ww[self.N*2:]
        subthreshold = self.design_matrix_proj(spk, W) + b[:,None] + U[:,None]*ipt.T
        ll = np.sum(spk * np.log(self.nonlinearity(subthreshold)+eps) \
                    - self.nonlinearity(subthreshold)*self.dt) \
                    - self.lamb*np.linalg.norm(W)
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
    
    def vec2param_kernel(self, ww, full_kernel=False):
        b = ww[:self.N]
        U = ww[self.N:self.N*2]
        W = ww[self.N*2:].reshape(self.N, self.N, self.nbasis)
        if full_kernel is False:
            return b, U, W
        elif full_kernel is True:
            Wijk = self.weight2kernel(weights=W)  # lesson: there is something about autograd array-box
            return b, U, Wijk

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
        
    def test_ll(self, spks):
        """
        After fitting parameters, we can use this fucntion to compute test log-likelihood
        """
        ll = 0
        return ll
        
    def log_marginal_kernel(self, ww, spks, ipts):
        """
        to avoid complextiy with states, use another function for kernel-based log-likelihood
        """
        ll = 0
        for spk,ipt in zip(spks, ipts):
            lli = -self.neg_log_likelihood_kernel(ww, spk.T, ipt=ipt)
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
    def fit_glm(self, data, num_iters=1000, optimizer="lbfgs", **kwargs):
        """
        Borrowing the ssm package method
        """
        optimizer = dict(adam=adam, bfgs=bfgs, rmsprop=rmsprop, lbfgs=lbfgs,
                         sgd=sgd)[optimizer]
        l_spk, l_ut = data
        if l_ut[0] is None and self.kernel_type!='basis':
            params = np.ones(self.N**2 + self.N)
        elif l_ut[0] is not None and self.kernel_type!='basis':
            params = np.ones(self.N**2 + self.N*2)
        elif l_ut[0] is not None and self.kernel_type=='basis':
            params = np.ones(self.N**2*self.nbasis + self.N*2)
        
        def _objective(params, itr):
            if self.kernel_type!='basis':
                obj = self.log_marginal(params, l_spk, l_ut)
            elif self.kernel_type=='basis':
                obj = self.log_marginal_kernel(params, l_spk, l_ut)
            return -obj

        params = optimizer(_objective,
                                params,
                                num_iters=num_iters,
                                **kwargs)
        if l_ut[0] is None and self.kernel_type!='basis':
            self.b, self.W = self.vec2param(params, state=False, ipt=False)
            self.U *= 0
        elif l_ut[0] is not None and self.kernel_type!='basis':
            self.b, self.U, self.W = self.vec2param(params)
        elif self.kernel_type=='basis':
            self.b, self.U, self.W = self.vec2param_kernel(params)
        
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
        
        