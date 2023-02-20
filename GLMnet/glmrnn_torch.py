#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:58:59 2023

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
# import glm_neuron as nonlinearity, kernel, spiking
import torch
import torch.nn as nn

class GLMRNN(nn.Module):
    
    def __init__(self,N, T, dt, k, kernel='basis', nl_type='exp', spk_type="bernoulli"):

        super(GLMRNN, self).__init__()
        
        self.N, self.T, self.dt, self.k, self.kernel, self.nl_type, self. spk_type = N,T,dt,k,kernel,nl_type,spk_type
        
        # future for class inheritance
        # self.glm = glm.GLM(self.n,self.d,self.c,observations=observations)
        
        ### GLM-RNN parameters
        self.W = nn.Parameter(torch.Tensor(N, N))     # connectivity matrix
        self.B = nn.Parameter(torch.Tensor(N))   # baseline firing
        self.tau = nn.Parameter(self.nonzero_time(torch.Tensor(1)))  # synaptic time constant
        nn.init.constant_(self.tau, 1)
        # self.tau = nn.Parameter(torch.Tensor(1))
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_(1/self.N)
            self.W.normal_(std=.1 / np.sqrt(self.N))
            
        ### taget parameters
        self.M = torch.randn(self.N)*1.
        self.b = 0
            
    def nonzero_time(self,x):
        relu = torch.nn.ReLU()
        nz = relu(x) + self.dt
        return nz
    
    def forward(self):
        """
        Forward process of the GLM-RNN model
        """
        xtt = torch.ones(self.N) #torch.zeros(self.N, self.T)  # voltage
        gtt = torch.ones(self.N) #torch.zeros(self.N, self.T)  # rate
        stt = torch.ones(self.N) #torch.zeros(self.N, self.T)  # spikes
        xt,gt,st = [xtt], [gtt], [stt]
        for t in range(self.T-1):
            xtt,gtt,stt = self.recurrence(xtt,gtt,stt)
            xt.append(xtt)  # append to list
            gt.append(gtt)
            st.append(stt)
        xt = torch.stack(xt, dim=0).T  # make to torch tensor
        gt = torch.stack(gt, dim=0).T
        st = torch.stack(st, dim=0).T
        return st, gt, xt
    
    def recurrence(self, xt,gt,st):
        """
        Recurrent activity used for RNN, in thise case for forward function and BPTT
        """
        xt_new = (1 - self.dt/self.tau)*xt + st  # synaptic acticity
        gt_new = self.nonlinearity(self.W @ xt_new + self.B)  # nonlinear rate
        try:
            st_new = self.spiking(gt_new*self.dt)  # spiking process
        except:
            print(self.W)
#        st_new = self.spiking(gt_new*self.dt)  # spiking process
        return xt_new, gt_new, st_new
    
    def generate_latent(self, latent_type=None):
        ### bistable latent example for now
        c = 0.  # posision
        sig = torch.zeros(1)+1.  # noise
        tau_l = 2  # time scale
        def vf(x):
            """
            Derivitive of focring in a double-well
            """
            return -(x**3 - 2*x - c)
        latent = torch.zeros(self.T)
        d = 1
        # self.M= torch.randn(self.N)  # loading matrix
        # self.b = 0
        for tt in range(self.T-1):
            ### simpe SDE
            latent[tt+1] = latent[tt] + self.dt/tau_l*(vf(latent[tt])) + torch.sqrt(self.dt*sig)*torch.randn(d)
        return latent
    
    def generate_target(self, latent, spk_type=None, nl_type=None, W_true=None):
        """
        Used to generate target spikes, given ground truth connectivity, phase-space of population spikes, or from latent dynamics
        """
        spk = torch.zeros(self.N, self.T)  # spike train
        rt = torch.zeros(self.N, self.T) # spike rate
        tau_r = torch.rand(self.N)*5
        for tt in range(self.T-1):
             spk[:,tt] = self.spiking(self.nonlinearity(self.M*latent[tt]-self.b))  # latent-driven spikes
             rt[:,tt+1] = rt[:,tt] + self.dt/tau_r*(-rt[:,tt] + spk[:,tt])
             
        return spk, rt  
        
    def nonlinearity(self, x, nl_type=None, add_param=None):
        if nl_type == 'exp' or nl_type == None:
            nl = torch.log(1+torch.exp(x)) #5/(1+torch.exp(x))+0  #torch.exp(x) # 
            
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
    
    def ll_loss(self, spk, rt):
        """
        log-likelihood loss function
        """
        eps = 1e-20
        ll = torch.sum(spk * torch.log((rt)+eps) - (rt)*self.dt)
        # ll = torch.sum(spk * torch.log(self.nonlinearity(self.W @ rt + self.B[:,None])) - self.nonlinearity(self.W @ rt + self.B[:,None])*self.dt)
        
        return -ll


### for MSE output loss
def mse_loss(outputs, targets):
    """
    MSE loss function of the full target spiking pattern
    """
    return torch.sum( (targets - outputs)**2) / outputs.shape[0]

# %% testing
N = 10
T = 1000
dt = 0.1
k = 10
my_glmrnn = GLMRNN(N, T, dt, k)
st, gt, xt = my_glmrnn.forward()
latent = my_glmrnn.generate_latent()

# %% training algorithm
n_epochs = 100
lr = 1e-3*.5
batch_size = 32
optimizer = torch.optim.Adam(my_glmrnn.parameters(), lr=lr)  # fancy gradient descent algorithm
losses = []


loss_npl = nn.PoissonNLLLoss()

for epoch in range(n_epochs):
    # latent = my_glmrnn.generate_latent()
    spk_target, gt_target = my_glmrnn.generate_target(latent)  # target spike patterns, given fixed latent
    st, gt, xt = my_glmrnn.forward()  # genertive model
    optimizer.zero_grad()
    
    # loss = loss_npl(torch.log(gt),spk_target)  # built in poisson loss
    loss = my_glmrnn.ll_loss(spk_target, gt)  # log-likelihood loss
    # loss = mse_loss(gt,gt_target)  # MSE loss
    
    loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
    optimizer.step()  # here it applies a step of gradient descent
    
    # with torch.no_grad():
    #     my_glmrnn.W[:] = my_glmrnn.W.clamp(-1, +1)
    #     my_glmrnn.B[:] = my_glmrnn.B.clamp(-2, +2)
    #     my_glmrnn.tau[:] = my_glmrnn.tau.clamp(dt,T)
    
    losses.append(loss.item())
    print(f'Epoch {epoch}, loss={loss:.3f}')
    loss.detach_()  # 2lines for pytorch administration
#    st.detach_()

# %% analysis
plt.figure()
plt.subplot(121)
plt.imshow(spk_target.detach().numpy(),aspect='auto')
st, gt, xt = my_glmrnn.forward()
plt.subplot(122)
plt.imshow(st.detach().numpy(),aspect='auto')