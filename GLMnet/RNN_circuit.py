# -*- coding: utf-8 -*-
"""
Created on Sat May  7 22:22:13 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.nn import init
from torch.nn import functional as F
import math

import matplotlib     
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %%
class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms. 
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()
        
    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 1
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.input2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        #######################################################
        #### extend with normalization and mask for h2h module!
        #######################################################

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.
        
        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)
        
        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
#        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = torch.tanh(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""
        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden

class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size
    
    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)
        
        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
    
# %%
batch_size = 10  # batch size
seq_len = 200    # sequence length
input_size = 2   # input dimension
output_size = 1  # output dimension

# Make some random inputs
input_rnn = torch.rand(seq_len, batch_size, input_size)

# Make network
rnn = RNNNet(input_size=input_size, hidden_size=200, output_size=2)

# Run the sequence through the network
out, rnn_output = rnn(input_rnn)

print('Input of shape (SeqLen, BatchSize, InputDim)=', input_rnn.shape)
print('Output of shape (SeqLen, BatchSize, Neuron)=', out.shape)

# %%
import torch.optim as optim
import time

# Instantiate the network and print information
hidden_size = 200  # network size
dt = 0.1          # time step
net = RNNNet(input_size=input_size, hidden_size=hidden_size,
             output_size=output_size, dt=dt)
print(net)

def train_model(net, dataset):
    """Simple helper function to train the model.
    
    Args:
        net: a pytorch nn.Module module
        dataset: a dataset object that when called produce a (input, target output) pair
    
    Returns:
        net: network object after training
    """
    # Use Adam optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.05)
    criterion = nn.MSELoss()
    #nn.CrossEntropyLoss()

    running_loss = 0
    running_acc = 0
    start_time = time.time()
    # Loop over training batches
    print('Training network...')
    for i in range(1000):#(1000):
        # Generate input and target, convert to pytorch tensor
        inputs, labels = dataset#()
        inputs = torch.from_numpy(inputs).type(torch.float)
        labels = torch.from_numpy(labels.flatten()).type(torch.float)
        #torch.from_numpy(labels.flatten()).type(torch.long)

        # boiler plate pytorch training:
        optimizer.zero_grad()   # zero the gradient buffers
        output, _ = net(inputs)
        # Reshape to (SeqLen x Batch, OutputSize)
        output = output.view(-1, output_size)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()    # Does the update

        # Compute the running loss every 100 steps
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}, Time {:0.1f}s'.format(
                i+1, running_loss, time.time() - start_time))
            running_loss = 0
    return net

#net = train_model(net, dataset)

# %%
# inputs, target = dataset()
# Input has shape (SeqLen, Batch, Dim) = (100, 16, 3)
# Target has shape (SeqLen, Batch) = (100, 16)
# %% simple chemotaxis dynamics
def vec2ang(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
#    dot_product = np.dot(unit_vector_1, unit_vector_2)
#    angle = np.arccos(dot_product)
    angle = math.atan2(unit_vector_2[1],unit_vector_2[0]) \
    - math.atan2(unit_vector_1[1],unit_vector_1[0])
    return angle
def ang2vec(vec,angle):
    oldX, oldY = vec[0], vec[1]
    newX = oldX * np.cos(angle) - oldY * np.sin(angle)
    newY = oldX * np.sin(angle) + oldY * np.cos(angle)
    return np.array([newX, newY])

K = 5 # angular noise
e1 = np.array([1,0])  #for a simple linear gradient along x axis!
targ = np.array([20,0])
rep = batch_size*1
T = seq_len*1
inputs = np.zeros((seq_len, batch_size, input_size))
target = np.zeros((seq_len, batch_size, output_size))
xss = target*0
for rr in range(rep):
    theta = np.zeros(T)
    dcs = theta*0
    dcps = theta*0
    xs = theta*0
    dths = theta*0
    theta[0] = np.random.vonmises(0,K)
    dths[0] = theta[0]
    vec = ang2vec(e1, theta[0])
    for tt in range(T-1):
        vec = ang2vec(vec, theta[tt])  # vector update according to track curvature
        bear = vec2ang(vec, targ)    # bearing angle
        xs[tt+1] = xs[tt] + vec[0]  # recording just the x-axis along gradient
        dcp = np.sin(bear+np.pi)  #tangential change
        wv = (dcp + np.random.vonmises(0,K))*dt
        dc = np.dot(vec, e1)*1  #parallel change
        P = 0.1/(1+np.exp(dc*dt))
        if np.random.rand() < P:
            beta = 1
        else:
            beta = 0
        rt = beta*np.random.vonmises(np.pi,K*0.2) 
        dth = wv + rt
        if dth > np.pi:
            dth = dth-2*np.pi
        if dth < -np.pi:
            dth = dth+2*np.pi
        theta[tt+1] = theta[tt] + dth*dt
        dths[tt+1] = dth
        dcs[tt+1] = dc
        dcps[tt+1] = dcp
    target[:,rr,0] = dths
    inputs[:,rr,0], inputs[:,rr,1] = dcs, dcps
    xss[:,rr,0] = xs
    
# %%
dataset = (inputs, target)
net = train_model(net, dataset)

# %%
activity_dict = {}  # recording activity
action_dict = {}    # recoding action
trial_infos = {}    # recording trial information

num_trial = 10
for i in range(num_trial):
    # Neurogym boiler plate
    # Sample a new trial
#    trial_info = env.new_trial()
    # Observation and groud-truth of this trial
#    ob, gt = env.ob, env.gt
    # Convert to numpy, add batch dimension to input
#    inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
    inputs_ = torch.from_numpy(inputs[:, i, :]).type(torch.float)
    
    # Run the network for one trial
    # inputs (SeqLen, Batch, InputSize)
    # action_pred (SeqLen, Batch, OutputSize)
    action_pred, rnn_activity = net(inputs_)
    
    # Compute performance
    # First convert back to numpy
    action_pred = action_pred.detach().numpy()[:, 0, :]
    action_dict[i] = action_pred
    # Read out final choice at last time step
#    choice = np.argmax(action_pred[-1, :])
    # Compare to ground truth
#    correct = choice == gt[-1]
    
    # Record activity, trial information, choice, correctness
    rnn_activity = rnn_activity[:, 0, :].detach().numpy()
    activity_dict[i] = rnn_activity
#    trial_infos[i] = trial_info  # trial_info is a dictionary
#    trial_infos[i].update({'correct': correct})

## Print informations for sample trials
#for i in range(5):
#    print('Trial ', i, trial_infos[i])
# %% performance
kk = 6
plt.figure()
plt.plot(target[:,kk])
plt.plot(action_dict[kk])

# %% network structure
plt.figure()
weight = net.rnn.h2h.weight.detach().numpy()
plt.imshow(weight, aspect='auto')


# %%
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_sigma = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
    def __init__(self, latent_size, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(latent_size, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        return torch.distributions.Normal(mu, torch.ones_like(mu))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z


transform = transforms.Compose(
    [transforms.ToTensor(),
     # Normalize the images to be -0.5, 0.5
     transforms.Normalize(0.5, 1)]
    )
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

input_dim = 28 * 28
batch_size = 128
num_epochs = 5
learning_rate = 0.001
hidden_size = 512
latent_size = 16

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=batch_size,
    shuffle=True, 
    pin_memory=torch.cuda.is_available())

print('Number of samples: ', len(mnist))

encoder = Encoder(input_dim, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_dim)

vae = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(
            q_z, 
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        loss = -(log_likelihood - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l, log_likelihood.item(), kl.item())
    
# %%
def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        inputs, _ = x,y
        x = inputs.view(-1, input_dim).to(device)
        z = autoencoder.encoder(x.to(device))
        z = z.rsample().detach().numpy() #.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
plot_latent(vae, dataloader)

# %%
class CTRNN2(nn.Module):

    def __init__(self, hidden_size, output_size, dt=None, **kwargs):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.tau = 1
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha

        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.lt = int(output_size/hidden_size)
        #######################################################
        #### extend with normalization and mask for h2h module!
        #######################################################

    def init_hidden(self, input_shape):
#        batch_size = input_shape[0]
#        return torch.zeros(batch_size, self.hidden_size)
        return torch.zeros(self.hidden_size)

    def recurrence(self, hidden):
        h_new = torch.tanh(self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):        
        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(0,self.lt)
        for i in steps:
            hidden = self.recurrence(hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden
    
# %% modify for time series
## neural dynamics N x T  #D_in: N*T
## netwok N x N  #latent: N*N
class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_sigma = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
#    def __init__(self, latent_size, H, D_out):
#        super(Decoder, self).__init__()
#        self.linear1 = torch.nn.Linear(latent_size, H)
#        self.linear2 = torch.nn.Linear(H, D_out)
#        
#
#    def forward(self, x):
#        x = F.relu(self.linear1(x))
#        mu = torch.tanh(self.linear2(x))
#        return torch.distributions.Normal(mu, torch.ones_like(mu))
    
    def __init__(self, latent_size, D_out, **kwargs):
        super().__init__()

        # Continuous time RNN
        J_dim = int(np.sqrt(latent_size))
        self.rnn = CTRNN2(J_dim, D_out, **kwargs)
        
        # Add an output layer
#        self.fc = nn.Linear(J_dim, D_out)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
#        x = self.fc(rnn_output)
#        x = F.relu(self.linear1(x))
#        mu = torch.tanh(self.linear2(x))
        mu = rnn_output.flatten()
        return torch.distributions.Normal(mu, torch.ones_like(mu))

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z
# %%
encoder_ = Encoder(input_dim, hidden_size, latent_size)
decoder_ = Decoder(latent_size, input_dim)
vae = VAE(encoder_, decoder_).to(device)
# %%
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs)
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()
        kl = torch.distributions.kl_divergence(
            q_z, 
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        loss = -(log_likelihood - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
    print(epoch, l, log_likelihood.item(), kl.item())