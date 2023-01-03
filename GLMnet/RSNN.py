# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 00:36:54 2023

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F

import matplotlib     
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %%
class RSNN(nn.Module , torch.autograd.Function):
    
    def __init__(self, input_dim, net_dim, output_dim, tau, dt, spk_param, init_std=1.):
        """
        Initialize an RSNN
        
        parameters:
        input_dim: int
        net_dim: int
        output_dim: int
        dt: float
        tau: float
        init_std: float, initialization variance for the connectivity matrix
        spk_param: three values for stochastic spiking process
        """
        # Setting some internal variables
        super(RSNN, self).__init__()  # pytorch administration line
        self.input_dim = input_dim
        self.N = net_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau = tau
        self.thr, self.temp, self.damp = spk_param  # spiking threshold, temperature, and dampled factor
        self.spike_op = self.Spike.apply  # snn-torch spike-opteration class
        
        # Defining the parameters of the network
        self.J = nn.Parameter(torch.Tensor(net_dim, net_dim))     # connectivity matrix
#        self.B = nn.Parameter(torch.Tensor(net_dim, input_dim))   # input weights
#        self.W = nn.Parameter(torch.Tensor(output_dim, net_dim))  # output matrix
        self.B = torch.Tensor(net_dim,input_dim)  # I/O without training constraint
        self.W = torch.Tensor(output_dim, net_dim)
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.N))
            self.W.normal_(std=1. / np.sqrt(self.N))
        
    def forward(self, inputs, initial_state=None):
        """
        Run the RSNN with input for one trial
        
        parameters:
        inp: torch tensor of shape (duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape ((duration+1) x net_size)
        output_seq: torch tensor of shape (duration x output_dim)
        """
#        # initialize arrays
#        T = inputs.shape[0]  # input time series
#        vt = torch.zeros(T+1 , self.N)  # voltage time series
#        zt = torch.zeros(T+1 , self.N)  # spiking time series
#        if initial_state is not None:
#            vt[0] = initial_state
#            zt[0] = self.spikefunction(vt[0])
#        output_seq = torch.zeros(T, self.output_dim)  # output time series
#        
#        # loop through time
#        for t in range(T):
#            vt[t+1,:] = (1 - self.dt/self.tau)*vt[t,:] + self.dt/self.tau*(self.J @ torch.sigmoid(vt[t,:]) + self.B @ inputs[t])
#            zt[t+1,:] = self.spikefunction(vt[t+1])
#            output_seq[t] = self.W @ self.NL(vt[t+1])
        
        # initialize arrays
        n_trials = inputs.shape[0]  # number of trials
        T = inputs.shape[1]  # input time series
        vt = torch.zeros((n_trials, T+1 , self.N))  # voltage time series
        zt = torch.zeros((n_trials, T+1 , self.N))  # spiking time series
        if initial_state is not None:
            vt[0] = initial_state
            zt[0] = self.spikefunction(vt[0])
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # output time series
        
        # loop through time
        for t in range(T):
            vt[:,t+1] = (1 - self.dt/self.tau)*vt[:,t] + self.dt/self.tau*(torch.sigmoid(vt[:,t]) @ self.J.T + inputs[:,t] @ self.B.T)
            zt[:,t+1] = self.spikefunction(vt[:,t+1])
            output_seq[:,t] = self.NL(vt[:,t+1]) @ self.W.T     
        
        self.save_for_backward(vt)  # test with this
        
        return vt, zt, output_seq
    
    @staticmethod
    def backward(self, grad_output):
        """
        Custom backward pass for RNN gradients
        """
        v_scaled, = self.saved_tensors
        dE_dz = grad_output*1
        dz_dv_scaled = self.pseudo_derivative(v_scaled)
        dE_dv_scaled = dE_dz * dz_dv_scaled
        return dE_dv_scaled
    
    def pseudo_derivative(self, v_scaled):
        """
        Bellec's pseudo-derivative for binary spikes
        """
        abs_u = torch.abs(v_scaled)
        return self.damp * torch.clamp(1-abs_u, min=0.0)

    def NL(self, x):
        """
        Synaptic nonliearity
        """
#        nl = torch.sigmoid(x)
        nl = torch.tanh(x)
        return nl
    
    def spikefunction(self, vt):
        """
        Stochastic spiking through Bernoulli process with rescaled/threshold voltage
        """
        new_v = (vt - self.thr)/self.thr
        new_z = torch.gt(torch.sigmoid(self.temp*new_v) , torch.rand(new_v.shape))
        return new_z
    
    @staticmethod
    class Spike(torch.autograd.Function):
        """
        Spiking opertation sub-class borrowed from snn-torch framework
        with this subclass, directly use Function.apply for spiking process 
        and the gradient step follows
        """
        @staticmethod
        def forward(ctx, v):
            spk = (v > 0).float() # Heaviside on the forward pass
            ctx.save_for_backward(spk)  # store the spike for use in the backward pass
            return
        @staticmethod
        def backward(ctx, grad_output):
            (spk,) = ctx.saved_tensors  # retrieve the spike
            grad = grad_output * spk # scale the gradient by the spike: 1/0
            return grad

def error_function(outputs, targets, masks):
    """
    parameters:
    outputs: torch tensor of shape (n_trials x duration x output_dim)
    targets: torch tensor of shape (n_trials x duration x output_dim)
    mask: torch tensor of shape (n_trials x duration x output_dim)
    
    returns: float
    """
    return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]


# %% test run
net_size = 100
dt = .1
tau = 1
spk_param = 0.4, .1 ,0.3  # threshold, temperature, damp
###  input_dim, net_dim, output_dim, tau, dt, spk_param, init_std=1.
my_net = RSNN(1, net_size, 1, tau, dt, spk_param, init_std=1.1)

# %% set simulation
# Let us run it with some constant input for a duration T=200 steps:
T = 200
n_trials = 100
inp = torch.zeros((n_trials, T, 1))  # the tensor containing inputs should have a shape (duration x input_dim), even if input_dim is 1.

v_seq, z_seq, output_seq = my_net.forward(inp, initial_state=torch.randn(net_size))  # this effectively runs the simulation

v_seq = v_seq.detach().squeeze().numpy()  # useful line for detaching the sequences from pytorch gradients (we will see that later)
output_seq = output_seq.detach().squeeze().numpy()
z_seq = z_seq.detach().squeeze().numpy()

# looking at the shapes of the obtained data
print(v_seq.shape)
print(output_seq.shape)

# %% test back-prop
n_epochs = 10
lr = 1e-3
batch_size = 32
n_trials = inputs.shape[0]
optimizer = torch.optim.Adam(my_net.parameters(), lr=lr)  # fancy gradient descent algorithm
losses = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    random_batch_idx = random.sample(range(n_trials), batch_size)
    batch = inputs[random_batch_idx]
    _, _, output = my_net.forward(batch)
    loss = error_function(output, targets[random_batch_idx], masks[random_batch_idx])
    loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
    optimizer.step()  # here it applies a step of gradient descent
    
    losses.append(loss.item())
    print(f'Epoch {epoch}, loss={loss:.3f}')
    loss.detach_()  # 2 lines for pytorch administration
    output.detach_()

# %%
inputs_pos, _, _, _ = generate_trials(100, coherences=[+1], T=T)
inputs_neg, _, _, _ = generate_trials(100, coherences=[-1], T=T)

# run network
v_pos, z_pos, output_pos = my_net.forward(inputs_pos*1)
v_neg, z_neg, output_neg = my_net.forward(inputs_neg)

# convert all tensors to numpy
v_pos = v_pos.detach().numpy().squeeze()
z_pos = z_pos.detach().numpy().squeeze()
v_neg = v_neg.detach().numpy().squeeze()
z_neg = z_neg.detach().numpy().squeeze()
output_pos = output_pos.detach().numpy().squeeze()
output_neg = output_neg.detach().numpy().squeeze()

plt.figure()
plt.plot(output_pos.T,'r')
plt.plot(output_neg.T,'b',alpha=0.1)

# %%
import random
def generate_trials(n_trials, coherences=[-2, -1, 1, 2], std=3., T=100):
    """
    Generate a set of trials for the noisy decision making task
    
    parameters:
    n_trials: int
    coherences: list of ints
    std: float, standard deviation of stimulus noise
    T: int, duration of trials
    
    returns (4-tuple):
    inputs: np array of shape (n_trials x T x input_dim)
    targets: np array of shape (n_trials x T x output_dim)
    masks: np array of shape (n_trials x T x 1)
    coherences: list of coherences chosen for each trial
    """
    inputs = std * torch.randn((n_trials, T, 1))
    targets = torch.zeros((n_trials, T, 1))
    mask = torch.zeros((n_trials, T, 1))
    mask[:, T-1] = 1  # set mask to one only at the end
    coh_trials = []
    
    for i in range(n_trials):
        coh = random.choice(coherences)  # choose a coherence
        inputs[i] += coh  # modify input
        targets[i, :] = 1 if coh > 0 else -1 # modify target
        coh_trials.append(coh)
        
    return inputs, targets, mask, coh_trials

inputs, targets, masks, coh_trials = generate_trials(100,T=T)
