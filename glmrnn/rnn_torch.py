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
        self.sig = nn.Parameter(torch.Tensor(N))   # noise strength within neuron
#        self.sig = nn.Parameter(torch.ones(N))
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.N))
            self.W.normal_(std=1. / np.sqrt(self.N))
            self.sig.uniform_(0.1, 5.0)
    
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
#            noise_t = torch.normal(torch.zeros(self.N), self.sig**2)
            x_seq[:, t+1] = (1 - self.deltaT) * x_seq[:, t] + self.deltaT * (torch.sigmoid(x_seq[:, t]) @ self.J.T  \
                          + inp[:, t] @ self.B.T) + self.deltaT**0.5 * (torch.randn_like(x_seq[:, t]) * self.sig**2)
            output_seq[:, t] = torch.sigmoid(x_seq[:, t+1]) @ self.W.T
        
        return x_seq, output_seq
    
# %% RNN with state readout

### RNN with state regularization and emissions ###
# the idea is to train RNN with GLM-HMM emisions
# this is deterministic RNN for now, but later can generalize to Poisson or other stochastic ones
# the higher-level goal is to analyze how the network represents state and transitions (birfurcation?)
###
        
class RNN_state(nn.Module):
    
    def __init__(self, input_dim, N, output_dim, state_dim, dt, init_std=1.):
        """
        Initialize an RNN
        
        parameters:
        input_dim: int
        N: int
        output_dim: int
        dt: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(RNN_state, self).__init__()  # pytorch administration line
        
        # Setting some internal variables
        self.input_dim = input_dim
        self.N = N
        self.output_dim = output_dim
        self.deltaT = dt
        self.state_dim = state_dim
        
        # Defining the parameters of the network
        self.B = nn.Parameter(torch.Tensor(N, input_dim))  # input weights
        self.J = nn.Parameter(torch.Tensor(N, N))   # connectivity matrix
        self.W = nn.Parameter(torch.Tensor(output_dim, N)) # output matrix
#        self.W = torch.eye(N)
        self.A = nn.Parameter(torch.Tensor(N, state_dim))  # state readout
        self.sig = nn.Parameter(torch.Tensor(N))   # noise strength within neuron
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.N))
            self.W.normal_(std=1. / np.sqrt(self.N))
#            self.W = torch.eye(N)
            self.A.normal_()
            self.sig.uniform_(0.1, 1.0)
    
    def forward(self, inp, initial_state=None):
        """
        Run the RNN with input for a batch of several trials
        
        parameters:
        inp: torch tensor of shape (n_trials x duration x input_dim)
        initial_state: None or torch tensor of shape (input_dim)
        
        returns:
        x_seq: sequence of voltages, torch tensor of shape (n_trials x (duration+1) x net_size)
        output_seq: torch tensor of shape (n_trials x duration x output_dim)
        state_seq: torch tensor of shape (n_trials x duration x state_dim)
        """
        n_trials = inp.shape[0]
        T = inp.shape[1]  # duration of the trial
        x_seq = torch.zeros((n_trials, T + 1, self.N)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # contains the sequence of output values z_{k, t} throughout the trial
        state_seq = torch.zeros((n_trials, T, self.state_dim))  # state output matrix
#        state_seq = torch.zeros((n_trials, T, 1))
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.deltaT) * x_seq[:, t] + self.deltaT * (torch.tanh(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T) \
                            + self.deltaT**0.5 *  (torch.randn_like(x_seq[:, t]) * self.sig**2)
            output_seq[:, t] = torch.tanh(x_seq[:, t+1]) @ self.W.T
            state_seq[:, t] = self.state( x_seq[:, t+1] @ self.A )
        
        return x_seq, output_seq, state_seq
    
    def state(self, xa):
        """
        input: 
        takes x @ A from the neural network as an input (state_dim)
        
        returns:
        one_hot vector that is biniary and is one at the max prob state
        """
        one_hot = torch.zeros_like(xa)
        exp_scores = torch.exp(xa)
        softmax_probs = exp_scores / torch.sum(exp_scores)
#        one_hot[torch.argmax(softmax_probs,1)] = 1
#        print(xa.shape)
        for ii in range(len(xa)):
            one_hot[ii,torch.argmax(softmax_probs[ii,:])] = 1
        
#        exp_scores = torch.exp(xa)
#        softmax_probs = exp_scores / torch.sum(exp_scores)
#        one_hot = torch.argmax(softmax_probs,1)[:,None]
        return one_hot
        
    
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
        self.sig = nn.Parameter(torch.Tensor(N))   # noise strength within neuron
        
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
            self.W = torch.eye(N)*1
            self.J.normal_(std=init_std / np.sqrt(self.N))
            self.sig.uniform_(0.1, 1.0)
            
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
        T = inp.shape[1]-0  # duration of the trial
        x_seq = torch.zeros((n_trials, T + 1, self.N)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T+0, self.N))  # fully readout rate
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.dt) * x_seq[:, t] + self.dt * \
            (torch.tanh(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T) \
            + self.dt**0.5*  (torch.randn_like(x_seq[:, t]) * self.sig**2)
            output_seq[:, t] = torch.tanh(x_seq[:, t+1]) @ self.W
        
        return x_seq, output_seq

# %% generic RNN trainer given class
class RNNTrainer():
    def __init__(self, RNN, loss_type, spk_target=None, st_target=None):
        self.rnn = RNN
        self.loss_type = loss_type
        self.spk_target = spk_target
        self.st_target = st_target
        self.alpha = 0.5  # 0-1 weight on state loss
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
                _, output, state_out = self.rnn.forward(batch)
                loss = self.error_function(output, targets[random_batch_idx], masks[random_batch_idx])*(1-self.alpha) \
                       + self.state_transition_loss(state_out, st_target[random_batch_idx])*self.alpha
#                       + self.state_transition_adhoc(st_target[random_batch_idx], state_out)

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
    
    def state_transition_loss(self, st, st_targ):
        """
        Cross entropy loss between state (onehot matrix) and the target
        """
        criterion = nn.CrossEntropyLoss()
        st_loss = criterion(st, st_targ.argmax(dim=1))  ###???
#        st_loss = criterion(st, st_targ)
        return st_loss
    
    def state_transition_adhoc(self, st, rt):
        """
        adding state-transition loss and weights to the objective function, with ad-hoc weights here
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
    
# %% other functions
def state2onehot(states, K=None):
    """
    state vector to onehot encoding, likely from numpy, used for state-constrained likelihood
    output onehot: trial x T x K
    """
    if K is None:
        nstate = torch.max(torch.from_numpy(states)) + 1
    else:
        nstate = K
    ntr, T = torch.Tensor(states).shape
    onehot = torch.zeros((ntr, T, nstate))
    for ii in range(ntr):
        for tt in range(T):
            onehot[ii, tt, int(states[ii,tt])] = 1
    return onehot