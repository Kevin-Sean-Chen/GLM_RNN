# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:41:49 2023

@author: kevin
"""

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.optim as optim

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

import ssm

# %%
# Generate synthetic time series data with input output relation, should generalize to probablistic ones later
per = 1.6
def generate_synthetic_IO_data(num_samples, sequence_length):
    data_in = []
    data_out = []
    for _ in range(num_samples):
        if np.random.rand() > 0.5:
            ipt_seq = 0.9 + np.random.randn(sequence_length, 1)*0.
            ipt_seq[-sequence_length//2:] = 0
            # ipt_seq[:10] = 0
            if np.random.rand()>0.999:
              opt_seq = (np.sin(np.linspace(0, 10, sequence_length)/per) + 1*np.random.normal(0, 0.1, sequence_length))[:,None]
              opt_seq[:25] = 0
            else:
              opt_seq = (-np.sin(np.linspace(0, 10, sequence_length)/per) + 1*np.random.normal(0, 0.1, sequence_length))[:,None]
              opt_seq[:25] = 0
        else:
            ipt_seq = -0.9 + np.random.randn(sequence_length, 1)*0.
            ipt_seq[-sequence_length//2:] = 0
            # ipt_seq[:10] = 0
            if np.random.rand()<0.999:
              opt_seq = (np.sin(np.linspace(0, 10, sequence_length)/per) + 1*np.random.normal(0, 0.1, sequence_length))[:,None]
              opt_seq[:25] = 0
            else:
              opt_seq = (-np.sin(np.linspace(0, 10, sequence_length)/per) + 1*np.random.normal(0, 0.1, sequence_length))[:,None]
              opt_seq[:25] = 0
        data_in.append(ipt_seq)
        data_out.append(opt_seq)
    return torch.tensor(data_in, dtype=torch.float32), torch.tensor(data_out, dtype=torch.float32)

aa,bb=generate_synthetic_IO_data(100,50)
plt.plot(bb.cpu().numpy()[:,:,0].T);

# %%
class CRNN(nn.Module):

    def __init__(self, input_dim, size, output_dim, deltaT, init_std=1.):
        """
        Initialize an CRNN

        parameters:
        input_dim: int
        size: int
        output_dim: int
        deltaT: float
        init_std: float, initialization variance for the connectivity matrix
        """
        super(CRNN, self).__init__()  # pytorch administration line

        # Setting some internal variables
        self.input_dim = input_dim
        self.size = size
        self.output_dim = output_dim
        self.deltaT = deltaT
        noise_scale = .1

        # Defining the parameters of the network
        self.B = nn.Parameter(torch.Tensor(size, input_dim))  # input weights
        self.J = nn.Parameter(torch.Tensor(size, size))   # connectivity matrix
        self.W = nn.Parameter(torch.Tensor(output_dim, size)) # output matrix
        # self.sig = nn.Parameter(torch.Tensor(size))   # noise strength within neuron
        self.sig = torch.rand(size)*noise_scale*1
        # self.SIG = nn.Parameter(torch.Tensor(size, size))

        # noise covariance
        matrix = torch.randn(size, size)
        covariance_matrix = (matrix + matrix.t()) / 2
        covariance_matrix = covariance_matrix @ covariance_matrix.T
        self.SIG = covariance_matrix*noise_scale

        # Initializing the parameters to some random values
        with torch.no_grad():  # this is to say that initialization will not be considered when computing the gradient later on
            self.B.normal_()
            self.J.normal_(std=init_std / np.sqrt(self.size))
            self.W.normal_(std=1. / np.sqrt(self.size))
            # self.sig.uniform_(0., noise_scale)#*noise_scale
            # self.SIG.normal_(std=init_std / np.sqrt(self.size))

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
        r_seq = x_seq*1
        samps = torch.distributions.MultivariateNormal(torch.zeros(self.size), self.SIG).sample((n_trials, T+1,))
        # by default the network starts with x_i=0 at time t=0 for all neurons
        if initial_state is not None:
            # x_seq[0] = initial_state
            x_seq[:,0,:] = initial_state
        output_seq = torch.zeros((n_trials, T+0, self.output_dim))  # contains the sequence of output values z_{k, t} throughout the trial

        # loop through time
        for t in range(T):
            ### continuous time
            x_seq[:, t+1] = (1 - self.deltaT) * x_seq[:, t] + self.deltaT * (self.nonlinearity(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T) \
                             + self.deltaT**0.5 *  (torch.randn_like(x_seq[:, t]) * self.sig**2) + samps[:,t]*1
            ### discrete time!
            # x_seq[:, t+1] = self.nonlinearity(x_seq[:, t]) @ self.J.T  + inp[:, t] @ self.B.T + samps[:,t]

            output_seq[:, t] = self.nonlinearity(x_seq[:, t+1]) @ self.W.T
            r_seq[:, t] = self.nonlinearity(x_seq[:, t+1])

        return x_seq, r_seq, output_seq

    def nonlinearity(self, x):
        nl = torch.tanh(x)
        # nl = 1/(1 + torch.exp(-x))
        return nl


def error_function(outputs, targets, masks=None):
    """
    parameters:
    outputs: torch tensor of shape (n_trials x duration x output_dim)
    targets: torch tensor of shape (n_trials x duration x output_dim)
    mask: torch tensor of shape (n_trials x duration x output_dim)

    returns: float
    """
    if masks is None:
      masks = torch.zeros_like(outputs) + 1
      masks[:,:25,:] = 0
    return torch.sum(masks * (targets - outputs)**2) / outputs.shape[0]

def batch_cov(xt):
    nb = len(xt)
    dJ = torch.cov(xt[0][-5:,:].T)
    for nn in range(1,nb):
      dJ = dJ + torch.cov(xt[nn][-5:,:].T)
    return dJ

def compute_covs(xt, rt, J):
    """
    Compute the esential covariances to update the noise covariance matrix
    input: xt (trial x time x neuron), rt, and matrix J
    output: new Sigma
    """
    T = xt.shape[1]
    xx = torch.einsum('bti,btj->bij', xt[:,1:,:], xt[:,1:,:])
    E_xx = torch.mean(xx, dim=0)
    xx = torch.einsum('bti,btj->bij', rt[:,:-1,:], xt[:,1:,:])
    E_rx = torch.mean(xx, dim=0)
    xx = torch.einsum('bti,btj->bij', xt[:,1:,:], rt[:,:-1,:])
    E_xr = torch.mean(xx, dim=0)
    xx = torch.einsum('bti,btj->bij', rt[:,:-1,:], rt[:,:-1,:])
    E_rr = torch.mean(xx, dim=0)
    new_Sig = 1/(T-1)*(E_xx - J @ E_rx - E_xr @ J.T + J @ E_rr @ J.T)
    return new_Sig

# %%
def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    """
    B = (A + A.T) / 2
    _, s, V = torch.svd(B)
    H = V.T @ torch.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    if isPD(A3):
        return A3
    spacing = np.spacing(torch.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.real(torch.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.cholesky(B)
        return True
    except torch.linalg.LinAlgError:
        return False

# %%
# Instantiate the models
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sequence_length = 50  # Length of each time series sequence
num_samples = 1000    # Number of time series sequences
input_size = 1#*sequence_length
hidden_size = 50
hidden_rec_size = 100
latent_size = 1*sequence_length  # focus on the linear readout for now, later move to the population activity...
output_size = 1
learning_rate = 0.001
num_epochs = 100
learning_rate_syn = .0051

# Generate synthetic data
dataI, dataO = generate_synthetic_IO_data(num_samples, sequence_length)#generate_synthetic_data(num_samples, sequence_length)
# train_loader = torch.utils.data.DataLoader(synthetic_data, batch_size=32, shuffle=True)

# generative_model = VanillaRNN(input_size, hidden_size, output_size).to(device)
deltaT = 0.1
generative_model = CRNN(input_size, hidden_size, output_size, deltaT, init_std=1.5)
#recognition_model = RecognitionNetwork(input_size, hidden_rec_size, latent_size).to(device)

# Define optimizers
gen_optimizer = optim.Adam(generative_model.parameters(), lr=learning_rate)
#rec_optimizer = optim.Adam(recognition_model.parameters(), lr=learning_rate)

batch_size = 50
xt_final = None

# Training loop
for epoch in range(1, num_epochs + 1):
    generative_model.train()
#    recognition_model.train()
    total_loss = 0.0
    # for batch_idx, (data_i,data_o) in enumerate(train_loader):
    random_batch_idx = random.sample(range(num_samples), batch_size)
    data_i = dataI[random_batch_idx].to(device)
    data_o = dataO[random_batch_idx].to(device)

        # # Recognition step
        # rec_optimizer.zero_grad()
        # mean, logvar = recognition_model(data)
        # z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)  # Reparameterization trick

        # # Generative step
        # gen_optimizer.zero_grad()
        # rnn_input = z.unsqueeze(1).expand(-1, sequence_length, latent_size)
        # output = generative_model(rnn_input)

        # control: only training rnn for now
    xt, rt, output = generative_model(data_i,xt_final)
    # xt, output = generative_model(data_i)
    with torch.no_grad():
        xt_final = xt[:,-1,:]*1

    # Compute VAE loss and backpropagation
    # gen_loss = vae_loss(output.squeeze(), data, mean, logvar)
    # gen_loss = nn.MSELoss()(output, data_o)
    # gen_loss = nn.CrossEntropyLoss()(output, data_o) ### play with this!
    gen_loss = error_function(output, data_o) + 0*torch.norm(generative_model.J, p='nuc')
    gen_loss.backward()
    gen_optimizer.step()

    # Manually update parameters (e.g., using gradient descent)
    with torch.no_grad():
        specific_param = getattr(generative_model, "J")
        ipt_param = getattr(generative_model, "B")
        specific_param += learning_rate_syn * batch_cov(generative_model.nonlinearity(xt[:,:10,:]))
        - 0.0*specific_param
        - 0. * torch.outer(getattr(generative_model, "B")[:,0],getattr(generative_model, "B")[:,0])  # effective like prior baseline?

    with torch.no_grad():
        J = getattr(generative_model, "J")
        Sig = compute_covs(xt, rt, J)
        Sig *= 1/len(random_batch_idx)
        generative_model.SIG = nearestPD(Sig)  #((Sig+Sig.T)/2)
        ### might want to add the regualirzed objective to paramerer loss function

    total_loss += gen_loss.item()

    # if batch_idx % log_interval == 0:
    #     print(f'Train Epoch: {epoch} [{batch_idx * len(data_i)}/{len(train_loader.dataset)}'
    #           f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {gen_loss.item():.6f}')

    print(f'Epoch {epoch}, Average Loss: {total_loss / len(output):.6f}')

# %%
# Load the trained model and set it to evaluation mode
generative_model.eval()

# Choose a batch of test data (you can adjust this as needed)
# test_batch = next(iter(train_loader))  # Change to test_loader if needed
# test_batch_out, test_batch_in = test_batch.to(device)
# test_batch_in, test_batch_out = test_batch.to(device)
num_test = 5
test_batch_in, test_batch_out = generate_synthetic_IO_data(num_test, sequence_length)

# Pass the test batch through the recognition model to get mean and logvar
# with torch.no_grad():
#     mean, logvar = recognition_model(test_batch)
#     z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)  # Reparameterization trick

# Pass the latent vectors through the generative model to get reconstructed outputs
with torch.no_grad():
    rnn_input = test_batch_in*1 #z.unsqueeze(1).expand(-1, sequence_length, latent_size)
    _,_,reconstructed_output = generative_model(rnn_input)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
for i in range(5):
    axes[0, i].plot(test_batch_out[i].cpu().numpy())
    axes[0, i].set_title('Original')

    axes[1, i].plot(reconstructed_output[i, :, 0].cpu().numpy())
    axes[1, i].set_title('Reconstructed')

plt.tight_layout()
plt.show()

# %%
### check the probability here
reps = 100
# rnn_input = dataI*1#torch.zeros((reps, sequence_length, 1))+0.9
rnn_input = torch.zeros((reps, sequence_length, 1))-0.9
rnn_input[:,25:,:] = 0
_,_,pred_out = generative_model(rnn_input)
plt.plot(pred_out.detach().numpy()[:,:,0].T);

frac = len(torch.where(pred_out[:,-1,:]>0)[0])/reps
print(frac)

#%%
### check with continuous task!
reps = 1000
rnn_input = torch.zeros((reps, sequence_length, 1))+0.3 #dataI*1#
# rnn_input = dataI*1#
# rnn_input[:,25:,:] = 0

rnn_input = dataI[:reps,:,:]*.9#
pred_out = rnn_input*0
plt.figure()
xt_f_pre = None
T_ = sequence_length+1
neuro_rec = torch.zeros((T_*reps,hidden_size))
stim_rec = torch.zeros(T_*reps)
choice_rec = torch.zeros(T_*reps)
for rr in range(reps):
    xt_pred, rt_pred, pred_out[rr,:,:] = generative_model(rnn_input[rr,:,:][None,:,:], xt_f_pre)
    xt_f_pre = xt_pred[:,-1,:]
    plt.plot(pred_out[rr,:,0].detach().numpy());
    ### record continuously
    neuro_rec[rr*T_:rr*T_+T_,:] = xt_pred[0,:,:]
    choice_rec[rr*T_:rr*T_+T_-1] = torch.squeeze(pred_out[rr,:,:])
    stim_rec[rr*T_:rr*T_+T_-1] = torch.squeeze(rnn_input[rr,:,:])


frac = len(torch.where(pred_out[:,-1,:]>0)[0])/reps
print(frac)

# %%
neuro_rec_ = neuro_rec.detach().numpy().T
stim_rec_ = stim_rec.detach().numpy()
choice_rec_ = choice_rec.detach().numpy()
wind = np.arange(0,5000)

plt.figure(figsize=(10, 7))
plt.imshow(neuro_rec_[:,wind], aspect='auto')
plt.xlabel('time', fontsize=30)
plt.ylabel('neuron', fontsize=30)
# plt.subplot(211)
# plt.plot(stim_rec.detach().numpy())
# plt.subplot(212)
# plt.plot(choice_rec.detach().numpy())
plt.figure(figsize=(15,10))
plt.plot(stim_rec_[wind], label='stim')
plt.plot(choice_rec_[wind]/1.5,'--', label='choice')
plt.xlabel('time', fontsize=30)
plt.legend(fontsize=30)

# %%
rec_c = []
rec_n = 0
for rr in range(reps):
    if dataI[rr,24,:]<0:
        rec_n += 1
        rec_c.append(pred_out[rr,-2,:]>0.)
print(sum(rec_c)/rec_n)

# %%
n_sess = 10
time_in_sess = reps//n_sess
ipt_ = rnn_input.reshape(n_sess, time_in_sess, sequence_length, input_size)
choice_ = pred_out.reshape(n_sess, time_in_sess ,sequence_length, output_size)
ipt_data = []
choice_data = []
for ss in range(n_sess):
    ipt_vec = np.zeros(time_in_sess)
    choice_vec = np.zeros(time_in_sess,dtype=int)
    for rr in range(time_in_sess):
        ### for inputs
        if ipt_[ss,rr,1,:]>0:
            ipt_vec[rr] = 1
        elif ipt_[ss,rr,1,:]<0:
            ipt_vec[rr] = -1
        ### for choices
        if choice_[ss,rr,-1,:]>0.5:
            choice_vec[rr] = int(1)
        elif choice_[ss,rr,-1,:]<0.5:
            choice_vec[rr] = int(0)
#    ipt_data.append(ipt_vec[:,None])
    ipt_data.append(np.concatenate((ipt_vec[:,None], np.ones((time_in_sess,1))),1))  # append for bias
    choice_data.append(choice_vec[:,None])

# %%
# Set the parameters of the GLM-HMM
num_states = 2        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 2         # input dimensions (weight and bias!)

# %% test
#num_sess = 20 # number of example sessions
#num_trials_per_sess = 100 # number of trials in a session
#inpts = np.ones((num_sess, num_trials_per_sess, input_dim)) # initialize inpts array
#stim_vals = [-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]
#inpts[:,:,0] = np.random.choice(stim_vals, (num_sess, num_trials_per_sess)) # generate random sequence of stimuli
#inpts = list(inpts) #convert inpts to correct format
#
## Generate a sequence of latents and choices for each session
#true_latents, true_choices = [], []
#for sess in range(num_sess):
#    true_z, true_y = true_glmhmm.sample(num_trials_per_sess, input=inpts[sess])
#    true_latents.append(true_z)
#    true_choices.append(true_y)

# %%
new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs", 
                   observation_kwargs=dict(C=num_categories), transitions="standard")

N_iters = 500 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(choice_data, inputs=ipt_data, method="em", num_iters=N_iters, tolerance=10**-4)

plt.plot(fit_ll)
# %%
posterior_probs = [new_glmhmm.expected_states(data=data, input=inpt)[0]
                for data, inpt
                in zip(choice_data, ipt_data)]

# %%
plt.figure()
plt.plot(posterior_probs[0])
plt.xlabel('trials', fontsize=30)
plt.ylabel('P(state)', fontsize=30)

plt.figure()
xtick_labels = ['wegith','bias']
plt.plot([0,1],new_glmhmm.observations.Wk.squeeze(),'-o')
plt.xticks([0,1], xtick_labels)
plt.ylabel('GLM weights', fontsize=30)
### Next steps:
# investigate noise and learning effects
# investigate neural dynamics
# add bias, history, and more stimuli tuning
# link to Poisson GLM in the other method
