#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:58:59 2023

@author: kschen
"""
import numpy as np
# import glm_neuron as nonlinearity, kernel, spiking
import torch
import torch.nn as nn

class GLMRNN(object):
    
    def __init__(self,N, T, dt, k, kernel='basis', nl_type='exp', spk_type="bernoulli"):

        super(GLMRNN, self).__init__()
        
        self.N, self.T, self.dt, self.k, self.kernel, self.nl_type, self. spk_type = N,T,dt,k,kernel,nl_type,spk_type
        
        # future for class inheritance
        # self.glm = glm.GLM(self.n,self.d,self.c,observations=observations)
        
        self.W = nn.Parameter(torch.Tensor(N, N))     # connectivity matrix
        self.B = torch.Tensor(N)  # baseline firing
        self.tau = nn.Parameter(self.nonzero_time(torch.Tensor(1)))  # synaptic time constant
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_(self.N)
            self.W.normal_(std=.1 / np.sqrt(self.N))
            
    def nonzero_time(self,x):
        relu = torch.nn.ReLU()
        nz = relu(x) + self.dt
        return nz
    
    def forward(self):
        """
        Forward process of the GLM-RNN model
        """
        xt = torch.zeros(self.N, self.T)  # voltage
        st = torch.zeros(self.N, self.T)  # spikes
        gt = torch.zeros(self.N, self.T)  # rate
        for t in range(self.T-1):
            xt[:,t+1] = (1 - self.dt/self.tau)*xt[:,t] + st[:,t]  # synaptic activity
            gt[:,t+1] = self.W @ xt[:,t+1] + self.B  # firing rate
            st[:,t+1] = self.spiking(self.nonlinearity(gt[:,t+1]*self.dt))  # spiking process
        return st, gt, xt
    
    
    def generate_target(self, spk_type=None, latent=None, W_true=None):
        """
        Used to generate target spikes, given ground truth connectivity, phase-space of population spikes, or from latent dynamics
        """
        ### bistable latent example for now
        c = 0.  # posision
        sig = torch.ones(1)+.5  # noise
        tau_l = 2  # time scale
        def vf(x):
            """
            Derivitive of focring in a double-well
            """
            return -(x**3 - 2*x - c)
        latent = torch.zeros(self.T)
        d = 1
        for tt in range(self.T-1):
            ### simpe SDE
            latent[tt+1] = latent[tt] + self.dt/tau_l*(vf(latent[tt])) + torch.sqrt(self.dt*sig)*torch.randn(d)
            
        ### simulate spikes    
        spk = torch.zeros(self.N, self.T)  # spike train
        rt = spk*1  # spike rate
        tau_r = torch.rand(self.N)*5
        M = torch.randn(self.N)  # loading matrix
        b = 0
        for tt in range(self.T-1):
             spk[:,tt] = self.spiking(self.nonlinearity(M*latent[tt]-b))  # latent-driven spikes
             rt[:,tt+1] = rt[:,tt] + self.dt/tau_r*(-rt[:,tt] + spk[:,tt])
             
        return spk
        
        
    def nonlinearity(self, x, nl_type=None, add_param=None):
        if nl_type == 'exp' or nl_type == None:
            nl = torch.exp(x)+1
            
        if nl_type=='ReLU':
            relu = torch.nn.ReLU()
            nl = relu(x)
            
        if nl_type=='loglin':
            nl = torch.log(1+torch.exp(x))

        if nl_type=='sigmoid':
            max_rate,min_rate = add_param
            nl = max_rate/(1+torch.exp(x)) + min_rate
        return nl

    def kernel(self, option, k):
        """
        Implement for future complext temporal kernels
        """
        k_vec = None
        return k_vec
    
    def spiking(self, lamb_dt, spk_type=None, add_param=None):
        if spk_type=='Poisson' or spk_type == None:
            spk = torch.poisson(lamb_dt)
            
        if spk_type=='Bernoulli':
            spk = torch.bernoulli(lamb_dt)
            
        if spk_type=='Neg-Bin':
            rnb = 0.2
            nb_dist = torch.distributions.negative_binomial.NegativeBinomial(lamb_dt+rnb, torch.zeros(len(lamb_dt))+rnb)
            spk = nb_dist.sample()
        return spk
    
def loss_function(outputs, targets, masks):
    """
    parameters:
    outputs: torch tensor of shape (n_trials x duration x output_dim)
    targets: torch tensor of shape (n_trials x duration x output_dim)
    mask: torch tensor of shape (n_trials x duration x output_dim)
    
    returns: float
    """
    return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]
# %% testing
N = 20
T = 1000
dt = 0.1
k = 10
my_glmrnn = GLMRNN(N, T, dt, k)

# %%
st, gt, xt = my_glmrnn.forward()