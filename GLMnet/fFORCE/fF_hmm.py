# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 00:42:02 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#%matplotlib qt5
# %% some params
nsecs = 200                       # time length
dt = 0.1                          # time steps
simtime = np.arange(0, nsecs, dt) # time vector
lt = len(simtime)                 # length of simulation  
N = 200                           # number of neurons
p = 0.2                           # sparsity of connection
g = 1.5                           # g greater than 1 leads to chaotic networks.
alpha = 1.                        #　learning initial constant
learn_every = 5                   # update every few steps

# %% RNN model
### RNN network
tau = 1  # membrane time scale
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask  # sparse chaotic network
M_d = M*1   # driven network
### readout weights
nRec2Out = N
wo = np.zeros((nRec2Out,1))+ np.random.randn(nRec2Out,1)*0.
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)*1
### driven weights
wo_d = np.random.randn(nRec2Out,1)
dw_d = np.zeros((nRec2Out,1))
wf_d = wf.copy()
### hint state weights
wh = 2.0*(np.random.rand(N,1)-0.5)*.2
z_state = np.random.randn(N,2)  #k-states

### for IRLS
P = (1.0/alpha)*np.eye(N)
P_d = P*1

### initialization
wo_len = np.zeros((1,lt))    
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,lt))
rt = np.zeros((N,lt))
xt_d = np.zeros((N,lt))
rt_d = np.zeros((N,lt))
zt = np.zeros((1,lt))
st = np.zeros(lt)

xt[:,-1][:,None] = x0
rt[:,-1][:,None] = np.tanh(x0)

# %% Markov model
k_states = 2
A = np.ones((k_states, k_states))
A = A / np.sum(A,0)
A = np.array([[0.99,0.01],[0.01,0.99]])  # stickier!
pi = np.ones(k_states)
pi = pi / np.sum(pi)

# %% target
###target pattern
amp = 1
freq = 1/20;
ft_ = amp*np.sin(np.pi*freq*simtime*.5)
high_f = amp*np.sin(np.pi*freq*simtime*5)
low_f = amp*np.sin(np.pi*freq*simtime*1)
offset = .0
pos_h = np.where(ft_>offset)[0]
pos_l = np.where(ft_<=offset)[0]
ft = np.zeros((2,lt))
ft[0,pos_h] = high_f[pos_h]*1.5
ft[1,pos_l] = low_f[pos_l]*1.2
#ft = ft -np.mean(ft,1)[:,None]#+ 0.
ft = ft.sum(0)
plt.figure()
plt.subplot(211)
plt.plot(ft.T)  #raw trace target  
SS = np.zeros((2,lt))-0
SS[0,pos_h] = 1
SS[1,pos_l] = 1
plt.subplot(212)
plt.plot(SS.T)  #underlying states

# %%
def stim_ssm2():
    # parameter
    a = 0.998  # Pr(X(t+1) = 0 | X(t) = 0)
    b = 0.998  # Pr(X(t+1) = 1 | X(t) = 1)
    P = np.array([[a, 1-a], [1-b, b]])  # transition matrix
    X = np.zeros(lt, dtype=int)  # Storage Matrix
    # Markov-chain
    X[0] = np.random.choice(2, 1, p=[0.5, 0.5])  # initialize
    for t in range(lt-1):  # loop over time within time series
        if X[t] == 0:
            X[t+1] = np.random.choice(2, 1, p=P[0, :])
        else:
            X[t+1] = np.random.choice(2, 1, p=P[1, :])
    # dynamics
    amp = 1
    freq = 1/20;
    high_f = amp*np.sin(np.pi*freq*simtime*1)
    low_f = amp*np.sin(np.pi*freq*simtime*5.)
    offset = .5
    pos_h = np.where(X>offset)[0]
    pos_l = np.where(X<offset)[0]
    ft = np.zeros((2,lt))
    ft[0,pos_h] = high_f[pos_h]*1.1
    ft[1,pos_l] = low_f[pos_l]*1.1
    #ft = ft -np.mean(ft,1)[:,None]#+ 0.
    zt = np.zeros((2,lt))
    zt[0,pos_h] = 1
    zt[1,pos_l] = 1
    return ft.sum(0), zt

#ft, SS = stim_ssm2()
#plt.figure()
#plt.subplot(211)
#plt.plot(ft)
#plt.subplot(212)
#plt.plot(SS.T)

# %% RNN　fucntions
def RNN(w, J, x_t, inpt):
    """
    Vanilla RNN update, with readout w, network J, voltage x, and input inpt
    """
    x = (1.0-dt/tau)*x_t + J @ (np.tanh(x_t)*dt)/tau + dt*inpt/tau
    r = np.tanh(x)
    z_hat = w.T @ r
    return z_hat, x, r

def RNN_loss(w, x_t, f_t):
    error = -0.5*(w.T @ np.tanh(x_t) - f_t)**2
    return error

def RLS(P, r):
    k = P @ r
    rPr = r.T @ k
    c = 1.0/(1. + rPr)
    P = 1/1*(P - k @ (k.T * c))  #projection matrix
    return P, k, c

# %% try iteration...
dl = 0
emt = -1
msteps = 5
aa = 0.2
llt = []
while emt<msteps:# and np.abs(dl)<10**-6:
    emt = emt+1
    ### new stimulus instantiation
#    ft, _ = stim_ssm2()
## %% E-step
    ### FB algorithm0
    ### Forward pass ###
    alphas = np.ones((k_states, lt))/k_states
    gammas = alphas*1
    p_y_x = alphas*1
    c = np.zeros(lt)*np.nan
    err_trace = np.zeros(k_states)
    
    for tt in range(lt):
        errs = np.zeros(k_states)
        for kk in range(k_states):
            ### target network
            z_hat, x_, r_ = RNN(wo, M, xt[:,tt-1][:,None], 0)
            ### input to drive another network
            inpt = wh*z_state[:,kk][:,None] + wf_d*ft[tt]  #state-input
            ### driven network
            z_d, x_d_, r_d_ = RNN(wo_d, M_d, xt_d[:,tt-1][:,None], inpt)
            ### record error
    #        errs[kk] = np.exp( RNN_loss(wo, xt_, ft[tt]) ) ### need to modify this...
            errs[kk] = np.mean(-0.5* (M @ r_ - M_d @ r_d_ - inpt)**2)
#            errs[kk] = np.mean(-0.5* (z_d - ft[tt])**2)
        errs = np.exp((errs)+np.log(gammas[:,tt]))  ### exponential here??
        errs = (1-aa)*errs + aa*err_trace
        err_trace = errs
#        errs = errs*gammas[:,tt]
        errs = errs/np.sum(errs)
        
        ### update with best state, and use FORCE learning! (M-step for RNN!)
        k_star = np.argmax(errs)
#        k_star = np.random.choice(np.arange(k_states), p=errs)
        st[tt] = k_star ### k_star, SS[0,tt], 0
        zt.T[tt][:,None], xt[:,tt][:,None], rt[:,tt][:,None] = RNN(wo, M, xt[:,tt-1][:,None], 0)
        inpt_star = wh*z_state[:,int(st[tt])][:,None] + wf_d*ft[tt]
        z_d, xt_d[:,tt][:,None], rt_d[:,tt][:,None] = RNN(wo_d, M_d, xt_d[:,tt-1][:,None], inpt_star)
        r, r_d = rt[:,tt][:,None], rt_d[:,tt][:,None]
        if np.mod(tt, learn_every) == 0:       
            ### RLS calculation
            P, k, cc = RLS(P, r)
            P_d, k_d, c_d = RLS(P_d, r_d)
            ### error terms
            e = zt.T[tt][:,None] - ft[tt]
            e_d = z_d - ft[tt]
            e_J = M @ r - M_d @ r_d - inpt_star
            ### update weights ########################### should update with weighted params! ###
            dw = -k*cc*e.T
            wo = wo + dw  # target readout
            dw_d = -k_d*c_d*((1*e_d).T)
            wo_d = wo_d + dw_d
            dJ = -e_J @ (P @ r).T
            M = M + dJ    # target network
        
        ### contiune with forward pass
        p_y_x[:,tt] = errs
        if tt == 0: ###
            alphas[:,tt] = pi * p_y_x[:,tt] 
        else:
            alphas[:,tt] = p_y_x[:,tt] * A.T @ alphas[:,tt-1]  #Bishop 13.36
        
        c[tt] = np.sum(alphas[:,tt])
        alphas[:,tt] = alphas[:,tt]/c[tt]  #Bishop 13.59
    
    ll = np.sum(np.log(c))
    ll_norm = np.exp(ll/lt)
    if emt==0:
        ll_t_ = ll_norm
    elif emt>0:
        dl = ll_norm-ll_t_
#        if dl<0:
#            break
        ll_t_ = ll_norm
    llt.append(ll_norm)
    
    ### Backward pass ###
    betas = alphas*0
    betas[:,-1] = np.ones(k_states)  # Bishop 13.39
    
    for tt in range(lt-1,0,-1):
        if tt == lt-1: ###
            betas[:,tt] = np.ones(k_states)
        else:
            betas[:,tt] = A @ (betas[:,tt+1] * p_y_x[:,tt+1])  #Bishop 13.38
            betas[:,tt] = betas[:,tt]/c[tt+1]   #Bishop 13.62
    
    
    ### posterior distribution ###
    gammas = alphas * betas  # E[z_t], Bishop 13.64
    xis = np.zeros((k_states, k_states, lt))
    for tt in range(lt):
        xis[:,:,tt] = np.outer(alphas[:,tt-1]*p_y_x[:,tt], betas[:,tt]) * A *1/c[tt]
        #(alphas(:,Ts(t)-1)*(py_z(:,Ts(t)).*betas(:,Ts(t)))').*model.A/c(Ts(t));
        
    ## %% M-step (for Markov model)
    temppi = np.mean(gammas,1)
    pi = temppi/np.sum(temppi)
    tempA = np.sum(xis, 2)
    A = tempA/np.sum(tempA,0)
    
    print(ll_norm)
    
# %% test
zpt = np.zeros((2,lt))
Pt_inf = zpt*0  #inferred hiddne state probability
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)*1-1):
    ti = ti+1 
    
    x = (1.0-dt/tau)*x + M @ (r*dt)/tau + 0*wf*z_state[:,int(SS[0,ti])][:,None]
    r = np.tanh(x)
    z = wo.T @ r

    zpt[:,ti] = z.squeeze()
    e = z - ft[ti]
       
zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft.T,'--',label='target')
plt.plot(zpt.T,label='readout')
