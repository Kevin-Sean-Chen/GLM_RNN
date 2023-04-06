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
import random

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

# %%
###############################################################################
# %% test with a task
###############################################################################

class RNN(nn.Module):
    
    def __init__(self, input_dim, size, output_dim, deltaT, init_std=1.):
        """
        Initialize an RNN
        
        parameters:
        input_dim: int
        size: int
        output_dim: int
        deltaT: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(RNN, self).__init__()  # pytorch administration line
        
        # Setting some internal variables
        self.input_dim = input_dim
        self.size = size
        self.output_dim = output_dim
        self.deltaT = deltaT
        
        # Defining the parameters of the network
        self.B = nn.Parameter(torch.Tensor(size, input_dim))  # input weights
        self.J = nn.Parameter(torch.Tensor(size, size))   # connectivity matrix
        self.W = nn.Parameter(torch.Tensor(output_dim, size)) # output matrix
        
        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.size))
            self.W.normal_(std=1. / np.sqrt(self.size))
            
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
        x_seq = torch.zeros((n_trials, T + 1, self.size)) # this will contain the sequence of voltage throughout the trial for the whole population
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            x_seq[0] = initial_state
        output_seq = torch.zeros((n_trials, T, self.output_dim))  # contains the sequence of output values z_{k, t} throughout the trial
        
        # loop through time
        for t in range(T):
            x_seq[:, t+1] = (1 - self.deltaT) * x_seq[:, t] + self.deltaT * (torch.sigmoid(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T)
            output_seq[:, t] = torch.sigmoid(x_seq[:, t+1]) @ self.W.T
        
        return x_seq, output_seq
        

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
        
# Complete the following function returning the error

def error_function(outputs, targets, masks):
    """
    parameters:
    outputs: torch tensor of shape (n_trials x duration x output_dim)
    targets: torch tensor of shape (n_trials x duration x output_dim)
    mask: torch tensor of shape (n_trials x duration x output_dim)
    
    returns: float
    """
    return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]

def train(net, inputs, targets, masks, n_epochs, lr, batch_size=32):
    n_trials = inputs.shape[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # fancy gradient descent algorithm
    losses = []
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        random_batch_idx = random.sample(range(n_trials), batch_size)
        batch = inputs[random_batch_idx]
        _, output = net.forward(batch)
        loss = error_function(output, targets[random_batch_idx], masks[random_batch_idx])
        loss.backward()  # with this function, pytorch computes the gradient of the loss with respect to all the parameters
        optimizer.step()  # here it applies a step of gradient descent
        
        losses.append(loss.item())
        print(f'Epoch {epoch}, loss={loss:.3f}')
        loss.detach_()  # 2 lines for pytorch administration
        output.detach_()
        
    return losses

# %%
inputs, targets, masks, coh_trials = generate_trials(200)

net_size = 50
deltaT = .2
my_net = RNN(1, net_size, 1, deltaT, 1.)
losses = train(my_net, inputs, targets, masks, 100, lr=1e-3)  # you have to find a good learning rate ! (try negative power of 10)

plt.plot(np.arange(len(losses)), losses)
plt.title('Learning curve')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %%
def accuracy(net):
   n_trials = 100
   inputs_pos, _, _, _ = generate_trials(n_trials, coherences=[+1])
   inputs_neg, _, _, _ = generate_trials(n_trials, coherences=[-1])
   _, outputs_pos = net.forward(inputs_pos)
   _, outputs_neg = net.forward(inputs_neg)
   outputs_pos = outputs_pos.detach().numpy().squeeze()
   outputs_neg = outputs_neg.detach().numpy().squeeze()
   acc_pos = np.sum(outputs_pos[:, -1] > 0) / 100
   acc_neg = np.sum(outputs_neg[:, -1] < 0) / 100
   return (acc_pos + acc_neg) / 2

print(accuracy(my_net))
# %%
inputs_pos, _, _, _ = generate_trials(100, coherences=[+1])
inputs_neg, _, _, _ = generate_trials(100, coherences=[-1])

# run network
traj_pos, output_pos = my_net.forward(inputs_pos)
traj_neg, output_neg = my_net.forward(inputs_neg)

# convert from voltages to firing rates
traj_pos = torch.tanh(traj_pos)
traj_neg = torch.tanh(traj_neg)

# convert all tensors to numpy
traj_pos = traj_pos.detach().numpy().squeeze()
output_pos = output_pos.detach().numpy().squeeze()
traj_neg = traj_neg.detach().numpy().squeeze()
output_neg = output_neg.detach().numpy().squeeze()

# condition averaging
mean_traj_pos = np.mean(traj_pos, axis=0)
mean_traj_neg = np.mean(traj_neg, axis=0)
mean_output_pos = np.mean(output_pos, axis=0)
mean_output_neg = np.mean(output_neg, axis=0)

plt.plot(mean_output_pos, c='r')
for i in range(output_pos.shape[0]):
    plt.plot(output_pos[i], c='r', alpha=.05)

plt.plot(mean_output_neg, c='b')
for i in range(output_neg.shape[0]):
    plt.plot(output_neg[i], c='b', alpha=.05)
    
plt.xlabel('timesteps')
plt.ylabel('network readout')

# %%
def plot_neuron_condition_averaged(neuron_idx, traj_pos, traj_neg, mean_traj_pos, mean_traj_neg):
    plt.plot(mean_traj_pos[:, neuron_idx], c='r')
    for i in range(traj_pos.shape[0]):
        plt.plot(traj_pos[i, :, neuron_idx], c='r', alpha=.05)

    plt.plot(mean_traj_neg[:, neuron_idx], c='b')
    for i in range(traj_neg.shape[0]):
        plt.plot(traj_neg[i, :, neuron_idx], c='b', alpha=.05)
    plt.xlabel('timesteps')
    plt.ylabel('firing rate')
    plt.title(f'Condition averaged responses for neuron {neuron_idx}')
    plt.show()
    
    
plot_neuron_condition_averaged(0, traj_pos, traj_neg, mean_traj_pos, mean_traj_neg)
plot_neuron_condition_averaged(1, traj_pos, traj_neg, mean_traj_pos, mean_traj_neg)
plot_neuron_condition_averaged(20, traj_pos, traj_neg, mean_traj_pos, mean_traj_neg)

# %% Test transfering to Poisson spiking network!
Jij = my_net.J.detach().numpy().squeeze()  #network
B = my_net.B.detach().numpy().squeeze()  #input
W = my_net.W.detach().numpy().squeeze()  #readout
dt = 0.1#my_net.deltaT
tau = 5
r_max = 10

def NL(x):
#    nl = np.log(1+np.exp(x))
    nl = r_max/(1+np.exp(-1*x)) + 0
    return nl
def spiking(nl):
    spk = np.random.poisson(nl)
    return spk
def glm_forward(inputs,J,B,W):
    """
    start with single trial input, with Tx1 dimension
    """
    lt = inputs.shape[0]
    N = J.shape[0]
    spk = np.zeros((N,lt))
    rt = np.zeros((N,lt))
    zt = np.zeros(lt)
    
    for tt in range(lt-1):
        spk[:,tt+1] = spiking( NL(J @ rt[:,tt] + B*inputs[tt])*dt )
        rt[:,tt+1] = rt[:,tt] + dt/tau*(-rt[:,tt] + tau*spk[:,tt])
        zt[tt+1] = W@rt[:,tt+1] #W@(NL(J @ rt[:,tt] + B*inputs[tt])*dt) #
    return spk, rt, zt

### cannot use tanh!!
ipt = inputs_pos[1,:,:].detach().numpy().squeeze()  # pick one input now
lamb = .1
spk,rt,zt = glm_forward(ipt, Jij*lamb,B,W*lamb)
plt.figure()
plt.imshow(spk, aspect='auto')
plt.figure()
plt.plot(zt)

# %%
def glm_forward_tense(inputs,J,B,W):
    """
    start with single trial input, with Tx1 dimension
    """
    ntrials = inputs.shape[0]
    lt = inputs.shape[1]
    N = J.shape[0]
    spk = np.zeros((ntrials, lt, N))
    rt = np.zeros((ntrials, lt, N))
    zt = np.zeros((ntrials, lt, inputs.shape[2]))
    
    for tt in range(lt-1):
        spk[:,tt+1] = spiking( NL(rt[:,tt]@J.T + inputs[:,tt]@B[:,None].T)*dt )
        rt[:,tt+1] = rt[:,tt] + dt/tau*(-rt[:,tt] + tau*spk[:,tt])
        zt[:,tt+1] = rt[:,tt+1]@W[None,:].T
    return spk, rt, zt

def accuracy_glm(net):
   n_trials = 100
   inputs_pos, _, _, _ = generate_trials(n_trials, coherences=[+1])
   inputs_neg, _, _, _ = generate_trials(n_trials, coherences=[-1])
   _, outputs_pos = net.forward(inputs_pos)
   _, outputs_neg = net.forward(inputs_neg)
   outputs_pos = outputs_pos.detach().numpy().squeeze()
   outputs_neg = outputs_neg.detach().numpy().squeeze()
   acc_pos = np.sum(outputs_pos[:, -1] > 0) / 100
   acc_neg = np.sum(outputs_neg[:, -1] < 0) / 100
   return (acc_pos + acc_neg) / 2

# %% scanning lambda and firing property
# record performance and spike counts with respect to model settings
lambs = np.array([0.01, 0.1, 1, 10])
rmaxs = np.array([1,5,10,50,100])
scan_spk = np.zeros((len(lambs), len(rmaxs)))
scan_per = scan_spk*0
n_trials = 100

for ll in range(len(lambs)):
    for rr in range(len(rmaxs)):
        inputs_pos, _, _, _ = generate_trials(n_trials, coherences=[+1])
        inputs_neg, _, _, _ = generate_trials(n_trials, coherences=[-1])
        inputs_pos = inputs_pos.detach().numpy()
        inputs_neg = inputs_neg.detach().numpy()
        r_max = rmaxs[rr]  # edit max firing rate
        spk_pos, _, zt_pos = glm_forward_tense(inputs_pos, Jij*lambs[ll], B, W*lambs[ll])
        spk_neg, _, zt_neg = glm_forward_tense(inputs_neg, Jij*lambs[ll], B, W*lambs[ll])
        acc_pos = np.sum(zt_pos[:, -1] > 0) / 100
        acc_neg = np.sum(zt_neg[:, -1] < 0) / 100
        scan_per[ll,rr] = (acc_pos + acc_neg) / 2
        scan_spk[ll,rr] = (spk_pos.mean() + spk_neg.mean()) / 2

# %%
plt.figure()
plt.imshow(scan_per, aspect='auto',interpolation='bicubic')
plt.yticks(np.arange(len(lambs)), lambs)
plt.xticks(np.arange(len(rmaxs)), rmaxs)
plt.colorbar()
plt.xlabel('max firing rate',fontsize=30)
plt.ylabel('rescale factor',fontsize=30)