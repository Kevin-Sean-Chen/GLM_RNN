# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 16:39:06 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

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
        # self.B = nn.Parameter(torch.Tensor(N, input_dim))  # input weights
        # self.W = nn.Parameter(torch.Tensor(output_dim, N)) # output matrix
        self.B = torch.randn(N, input_dim)
        self.W = torch.eye(N)  # identity readout for fully-observed network
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.W.normal_(std=1. / np.sqrt(self.N))
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
    