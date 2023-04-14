# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:39:06 2023

@author: kevin
"""

import numpy as np
#from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import random

# %% vanila RNN

class RNN(nn.Module):
    
    def __init__(self, input_dim, N, output_dim, dt, init_std=1.):
        """
        Initialize an RNN
        
        parameters:
        input_dim: int
        N: int
        output_dim: int
        dt: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(RNN, self).__init__()  # pytorch administration line
        
        # Setting some internal variables
        self.input_dim = input_dim
        self.N = N
        self.output_dim = output_dim
        self.deltaT = dt
        
        # Defining the parameters of the network
        self.B = nn.Parameter(torch.Tensor(N, input_dim))  # input weights
        self.J = nn.Parameter(torch.Tensor(N, N))   # connectivity matrix
        self.W = nn.Parameter(torch.Tensor(output_dim, N)) # output matrix
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.N))
            self.W.normal_(std=1. / np.sqrt(self.N))
    
    def forward(self, inp, initial_state=None):
        """
        Run the RNN with input for a batch of several trials
        
        parameters:
        inp: torch tensor of shape (n_trials x duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape (n_trials x (duration+1) x net_size)
        output_seq: torch tensor of shape (n_trials x duration x output_dim)
        """
        n_trials = inp.shape[0]
        T = inp.shape[1]  # duration of the trial
        x_seq = torch.zeros((n_trials, T + 1, self.N)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # contains the sequence of output values z_{k, t} throughout the trial
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.dt) * x_seq[:, t] + self.dt * (torch.sigmoid(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T)
            output_seq[:, t] = torch.sigmoid(x_seq[:, t+1]) @ self.W.T
        
        return x_seq, output_seq
    
# %% low-rank RNN
class lowrank_RNN(nn.Module):
    
    def __init__(self, input_dim, N, r, output_dim, dt, init_std=1.):
        """
        Initialize an RNN
        
        parameters:
        input_dim: int
        N: int
        r: int
        output_dim: int
        dt: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(lowrank_RNN, self).__init__()  # pytorch administration line
        
        # Setting some internal variables
        self.input_dim = input_dim
        self.N = N
        self.r = r
        self.output_dim = output_dim
        self.dt = dt
        
        # Defining the parameters of the network
        self.B = nn.Parameter(torch.Tensor(N, input_dim))  # input weights
        self.W = nn.Parameter(torch.Tensor(output_dim, N)) # output matrix
#        self.B = torch.zeros(N, input_dim, requires_grad=False)
#        self.W = torch.zeros(output_dim, N, requires_grad=False)
        self.m = nn.Parameter(torch.Tensor(N, r))  # left low-rank
        self.n = nn.Parameter(torch.Tensor(N, r))  # right low-rank
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.W.normal_(std=1. / np.sqrt(self.N))
            self.m.normal_()
            self.n.normal_()
            
    def forward(self, inp, initial_state=None):
        """
        Run the RNN with input for a batch of several trials
        
        parameters:
        inp: torch tensor of shape (n_trials x duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape (n_trials x (duration+1) x net_size)
        output_seq: torch tensor of shape (n_trials x duration x output_dim)
        
        """
#        self.m.data.clamp_(min=0.0)  # test with clamping
        
        n_trials = inp.shape[0]
        T = inp.shape[1]  # duration of the trial
        x_seq = torch.zeros((n_trials, T + 1, self.N)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # contains the sequence of output values z_{k, t} throughout the trial
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.dt) * x_seq[:, t] + self.dt * \
            (torch.sigmoid(x_seq[:, t]) @ (self.m @ self.n.T/self.N).T  + inp[:, t] @ self.B.T)
            output_seq[:, t] = torch.sigmoid(x_seq[:, t+1]) @ self.W.T
        
        return x_seq, output_seq
    
    def _mn2J(self):
        return (self.m @ self.n.T/self.N)
    
    def clamp(self):
        self.m = torch.clamp(self.m, min=0.0)
    
# %% readout RNN
class observed_RNN(nn.Module):
    
    def __init__(self, input_dim, N, dt, init_std=1.):
        """
        Initialize an RNN
        
        parameters:
        input_dim: int
        N: int
        r: int
        output_dim: int
        dt: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(observed_RNN, self).__init__()  # pytorch administration line
        
        # Setting some internal variables
        self.input_dim = input_dim
        self.N = N
        self.output_dim = N
        self.dt = dt
        
        # Defining the parameters of the network
        self.J = nn.Parameter(torch.Tensor(N, N))   # connectivity matrix
        self.B = nn.Parameter(torch.Tensor(N, input_dim))  # input weights
        # self.W = nn.Parameter(torch.Tensor(output_dim, N)) # output matrix
#        self.B = torch.randn(N, input_dim)
        self.W = torch.eye(N)  # identity readout for fully-observed network
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
#            self.W.normal_(std=1. / np.sqrt(self.N))
            self.W = torch.eye(N)
            self.J.normal_(std=init_std / np.sqrt(self.N))
            
    def forward(self, inp, initial_state=None):
        """
        Run the RNN with input for a batch of several trials
        
        parameters:
        inp: torch tensor of shape (n_trials x duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape (n_trials x (duration+1) x net_size)
        output_seq: torch tensor of shape (n_trials x duration x output_dim)
        
        """
     
        n_trials = inp.shape[0]
        T = inp.shape[1]  # duration of the trial
        x_seq = torch.zeros((n_trials, T + 1, self.N)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T+1, self.N))  # fully readout rate
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.dt) * x_seq[:, t] + self.dt * \
            (torch.sigmoid(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T)
            output_seq[:, t+1] = torch.sigmoid(x_seq[:, t+1]) @ self.W
        
        return x_seq, output_seq

# %% generic RNN trainer given class
class RNNTrainer():
    def __init__(self, RNN, loss_type, spk_target=None, st_target=None):
        self.rnn = RNN
        self.loss_type = loss_type
        self.spk_target = spk_target
        self.st_target = st_target
        if st_target is not None:
            K = st_target.shape[-1]  # K target states (st is true state time series trial x T x K)
            self.wk = nn.Parameter(torch.Tensor(RNN.N, K))

    def train(self, inputs, targets, masks, n_epochs, lr, batch_size=32):
        n_trials = inputs.shape[0]
        losses = []
        
        if self.loss_type == 'MSE':
            optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)  # fancy gradient descent algorithm
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                random_batch_idx = random.sample(range(n_trials), batch_size)
                batch = inputs[random_batch_idx]
                _, output = self.rnn.forward(batch)
                loss = self.error_function(output, targets[random_batch_idx], masks[random_batch_idx])
                loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
                optimizer.step()  # here it applies a step of gradient descent
                
                losses.append(loss.item())
                print(f'Epoch {epoch}, loss={loss:.3f}')
                loss.detach_()  # 2 lines for pytorch administration
                output.detach_()
                
        elif self.loss_type == 'joint':
            optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr)  # fancy gradient descent algorithm
            spk_target = self.spk_target
            loss_fn = nn.PoissonNLLLoss()
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                random_batch_idx = random.sample(range(n_trials), batch_size)
                batch = inputs[random_batch_idx]
                _, output = self.rnn.forward(batch)
                loss = self.error_function(output, targets[random_batch_idx], masks[random_batch_idx]) \
                       + loss_fn((targets[random_batch_idx]),  spk_target[random_batch_idx]+1e-10)
#                       + self.ll_loss(spk_target[random_batch_idx], targets[random_batch_idx])

                loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
                optimizer.step()  # here it applies a step of gradient descent
                
                losses.append(loss.item())
                print(f'Epoch {epoch}, loss={loss:.3f}')
                loss.detach_()  # 2 lines for pytorch administration
                output.detach_()
                
        elif self.loss_type == 'state':
            optimizer = torch.optim.Adam([{'params': self.rnn.parameters()}, {'params': self.wk}], lr=lr)
            st_target = self.st_target
            
            for epoch in range(n_epochs):
                optimizer.zero_grad()
                random_batch_idx = random.sample(range(n_trials), batch_size)
                batch = inputs[random_batch_idx]
                _, output = self.rnn.forward(batch)
                loss = self.error_function(output, targets[random_batch_idx], masks[random_batch_idx]) \
                       + self.state_transition(st_target[random_batch_idx], output)

                loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
                optimizer.step()  # here it applies a step of gradient descent
                
                losses.append(loss.item())
                print(f'Epoch {epoch}, loss={loss:.3f}')
                loss.detach_()  # 2 lines for pytorch administration
                output.detach_()
            
        return losses
    
    def error_function(self, outputs, targets, masks):
        """
        parameters:
        outputs: torch tensor of shape (n_trials x duration x output_dim)
        targets: torch tensor of shape (n_trials x duration x output_dim)
        mask: torch tensor of shape (n_trials x duration x output_dim)
        
        returns: float
        """
        return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]
    
    def ll_loss(self, spk, rt):
        """
        log-likelihood loss function
        """
        eps = 1e-20
        ll = torch.sum(spk * torch.log((rt)+eps) - (rt)*self.rnn.dt)
        # ll = torch.sum(spk * torch.log(self.nonlinearity(self.W @ rt + self.B[:,None])) - self.nonlinearity(self.W @ rt + self.B[:,None])*self.dt)
        return -ll
    
    def state_transition(self, st, rt):
        """
        adding state-transition loss and weights to the objective function
        st: trial x T x K
        rt: trial x T x N
        wk: N x k
        """
#        K = st.shape[0]  # k target states (st is true state time series k x T)
#        self.wk = nn.Parameter(torch.Tensor(self.N, K))  # weights across states 
        unormed_p = rt @ self.wk
        logp = unormed_p - torch.logsumexp(unormed_p, dim=2)[:,:,None]
        loss = - torch.sum(st * logp)
        return loss