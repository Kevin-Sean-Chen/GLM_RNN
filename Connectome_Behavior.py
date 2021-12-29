#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:55:46 2021

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import pandas as pd

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

# %% load connectome
file = r'/home/kschen/Downloads/NeuronConnect.xls'
df = pd.read_excel(file)
neuron_i = df['Neuron 1'].to_numpy()  #pre
neuron_j = df['Neuron 2'].to_numpy()  #post
syn_type = df['Type'].to_numpy()  #synapse
syn_number = df['Nbr'].to_numpy()  #number of connections

# %%
print('types of synapse: ', np.unique(syn_type))
# %% build adjacent matrix
neurons = np.unique(np.concatenate((neuron_i,neuron_j)))
nn = len(neurons)  #number of cells
ns = len(neuron_i)  #number of connetions
J = np.zeros((nn,nn))  #connectivity
J_s = J*0  #chemical synapse
J_g = J*0  #gap jucntion
J_nmj = J*0  # neuromuscular junction
for ii in range(ns):
    pre = neuron_i[ii]
    post = neuron_j[ii]
    sn = syn_number[ii]
    Ji = np.where(neurons==pre)[0]
    Jj = np.where(neurons==post)[0]
    J[Ji,Jj] = sn
    if len(np.where(syn_type[ii]==np.array(['R','Rp','S','Sp']))[0])>0:
        J_s[Ji,Jj] = sn
    elif syn_type[ii]=='EJ':
        J_g[Ji,Jj] = sn
    elif syn_type[ii]=='NMJ':
        # print(sn)
        J_nmj[Ji,Jj] = sn

# %%
plt.figure()
plt.imshow(J_s, aspect='auto')
#%% binary view
A = J*0
A[J_s!=0] = 1
plt.figure()
plt.imshow(A, aspect='auto')

###############################################################################
# %% attempt for modeling
###############################################################################
# %% connectome dynamics
# %%
### time structure
dt = 0.1
T = 200
time = np.arange(0,T,dt)
lt = len(time)
def sigmoid(x):
    # r = 1/(1+np.exp(-x))
    r = np.tanh(x)
    return r

### single cell property
tau = 2  #time constant
D = .5  #noise strength
pi = 0.3  #portion of inhibition
gg = 1.5

### synaptic statistics
J_dale = J*0+1
prob_inh = np.random.rand(nn)
J_dale[prob_inh<pi,:] = -1
J_wei = np.random.randn(nn,nn)*gg/np.sqrt(nn*pi)

### network dynamics
Vs = np.zeros((nn,lt))  #voltage traces
for tt in range(lt-1):
    vi, vj = np.meshgrid(Vs[:,tt], Vs[:,tt])
    Vs[:,tt+1] = Vs[:,tt] + dt/tau*(-Vs[:,tt] + 10.*(J_s*J_dale*J_wei) @ sigmoid(Vs[:,tt])\
                                    - .1*(J_g*(vi-vj)).sum(0)) + np.sqrt(dt*D)*np.random.randn(nn)
    
# %%
plt.figure()
plt.subplot(211)
plt.imshow(Vs,aspect='auto')
plt.subplot(212)
plt.plot(Vs.T)

# %% Behavioral dynamics
# %%
### kinematics
vv = 0.022  #cm/s
Dx = 0.1  #behavioral noise

### readout
w_11 = np.where(neurons=='SMBDL')[0][0] 
w_12 = np.where(neurons=='SMBDR')[0][0]
w_21 = np.where(neurons=='SMBVL')[0][0]
w_22 = np.where(neurons=='SMBVR')[0][0]
w_nmj = np.array([0.5,0.5])  #NMJ readout linear

### behavioral dynamics
xys = np.zeros((2,lt))  #x-y 2D
thetas = np.zeros(lt)  #head angles
for tt in range(lt-1):
    xys[:,tt+1] = xys[:,tt] + dt*( np.array([vv*np.cos(thetas[tt]), vv*np.sin(thetas[tt])]) )
    thetas[tt+1] = thetas[tt] + dt*( w_nmj[0]*(Vs[w_11,tt]-Vs[w_21,tt]) + w_nmj[1]*(Vs[w_12,tt]-Vs[w_22,tt]) )    

# %%
plt.figure()
plt.subplot(211)
plt.plot(xys[0,:], xys[1,:])
plt.subplot(212)
plt.plot(Vs[np.array([w_11,w_12,w_21,w_22]),:].T)
