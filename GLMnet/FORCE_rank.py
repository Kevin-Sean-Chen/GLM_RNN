# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 00:50:47 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#%matplotlib qt5
# %% FORCE functions
def Lorenz_dt(xyz, dt):
    ### Lorenz attractor in 3-D with parameters from the paper
    x, y, z = xyz
    dx = (10*(y-x))*dt
    dy = (x*(28-z)-y)*dt
    dz = (x*y-(8/3)*z)*dt
    return dx,dy,dz

def Lorenz_model(T,dt):
    ### Samples with ran length T and time step dt
    lt = int(T/dt)
    Xt = np.zeros((lt,3))
    Xt[0,:] = np.random.randn(3)*10  #initial condition within the dynamic range
    for tt in range(lt-1):
        dx,dy,dz = Lorenz_dt(Xt[tt,:], dt)
        Xt[tt+1,:] = Xt[tt,:]+[dx,dy,dz]
    return Xt

def FORCE_learning(M, ft):
    ### initialize parameters
    N = M.shape[0]
    T = len(ft)
    wo_len = np.zeros((1,T))    
    zt = np.zeros((1,T))
    x0 = 0.5*np.random.randn(N,1)
    z0 = 0.5*np.random.randn(1)
    rt = np.zeros((N,T))
    x = x0
    r = np.tanh(x)
    z = z0
    wo = np.zeros((N,1))
    
    ### learning dynamics
    ti = 0
    P = (1.0/alpha)*np.eye(N)
    for t in range(T-1):
        ti = ti+1
        x = (1.0-dt)*x + M @ (r*dt)   # RNN dynamics
        r = np.tanh(x)
        rt[:,t] = r[:,0]
        z = wo.T @ r   # linear reconstruction
        
        if np.mod(ti, learn_every) == 0:
            k = P @ r;
            rPr = r.T @ k
            c = 1.0/(1.0 + rPr)
            P = P - k @ (k.T * c)  #projection matrix
            
            # update the error for the linear readout
            e = z-ft[ti]
            
            # update the output weights
            dw = -e*k*c
            wo = wo + dw
            
            # update the internal weight matrix using the output's error
            M = M + np.repeat(dw,N,1).T          
    
        # Store the output of the system.
        zt[0,ti] = np.squeeze(z)
        wo_len[0,ti] = np.sqrt(wo.T @ wo)	
        
    return rt, zt, M   #dynamics, reconstruction, and network

def G_FORCE_learning(M_, ft):
    ### initialize parameters
    N = M_.shape[0]
    T = len(ft)
    wo_len = np.zeros(T)
    wo = np.ones((N,1))
    zt = np.zeros(T)
    x0 = 0.5*np.random.randn(N)
    z0 = 0.5*np.random.randn(1)
    xt = np.zeros((N,T))
    rt = np.random.randn(N,T)
    spks = np.zeros((N,T))
    xt[:,0] = x0
    z = z0

    P = (1.0/alpha)*np.eye(N)
    for tt in range(1, T):
        #GLM-RNN
        rec_pot = (1-dt/tau_r)*rt[:,tt-1] + spks[:,tt-1]*dt/tau_r  #slow recover
        rt[:,tt] = rec_pot
        spk = spiking( (M_ @ rec_pot) , dt)  #generate spike s with current u
        spks[:,tt] = spk
        
        #reconstruct dynamics
        z = wo.T @ rec_pot #tens
        
        #learning
        if np.mod(tt, learn_every) == 0:
            dr = (rec_pot)  #taking derivative over the nonlinearity #deNL
            k = (P @ dr)[:,None]
            rPr = dr[:,None].T @ k
            c = 1.0/(1.0 + rPr)
            P = P - (k @ k.T) * c  #projection matrix
        	
            # update the error for the linear readout
            e = z-ft[tt] ### how is error computed!
    	
        	# update the output weights
            dw = -(e[:,None]*k*c)#[:,None]
            wo = wo + dw
            
            # update the internal weight matrix using the output's error
            M_ = M_ + (E @ dw.T) #np.repeat(dw,N,1).T  #
         
        #print(tt,z)
        # Store the output of the system.
        zt[tt] = np.squeeze(z)
        wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))
    
    return spks, rt, zt, M_   # spikes, rate, reconstruction, and network

def spiking(x,dt):

#    x = 1/gain*np.log(1+np.exp(gain*x))
    x[x<0] = 0
    spike = np.random.poisson(x*dt*gain)
    return spike

# %% FORCE parameters
N = 300  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0  #learning initial constant
dt = 0.1
nsecs = 500
learn_every = 2  #effective learning rate
scale = 1.0/np.sqrt(p*N)  #scaling connectivity

###target pattern
simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 2
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
ft = ft/1.5
#ft = Lorenz_model( nsecs/30 , dt/30)/30
#ft = ft[:,0]

# %% scanning network rank
cuts = np.array([5,10,50,100,200,299])
MSEs = np.zeros(len(cuts))
times_mle = np.zeros(len(cuts))

for cc in range(len(cuts)):
    
    M = np.random.randn(N,N)*g*scale
    uu,ss,vv = np.linalg.svd(M)
    ss[cuts[cc]:] = 0
    M = uu @ np.diag(ss) @ vv
    sparse = np.random.rand(N,N)
    mask = np.random.rand(N,N)
    mask[sparse>p] = 0
    mask[sparse<=p] = 1
    M = M*mask
    rt, zt, M = FORCE_learning(M, ft)
    zt = np.squeeze(zt)
    MSEs[cc] = sum((zt-ft)**2)/sum(zt**2)
    
# %%
plt.figure()
plt.plot(cuts, MSEs, '-o',label='FORCE')
plt.xlabel('initial rank',fontsize=40)
plt.ylabel('normalized MSE',fontsize=40)

plt.plot(cuts, np.array([8,6.6,0.17,0.16,0.11,0.03]), '-o', label='G-FORCE')
plt.legend(fontsize=40)

###############################################################################
# %%
###############################################################################
# %% G-FORCE linear setup
#size and length
N = 300
T = 500
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate
#network parameters
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = 1.
E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity

#input parameters
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 2
ft = 1*(amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    1*(amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    1*(amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    1*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
ft = ft*100
#ft = Lorenz_model( nsecs/30 , dt/30)*5
#ft = ft[:,0]


# %% scanning network rank
cuts = np.array([1,5,10,50,100,200,299])
MSEs_glm = np.zeros(len(cuts))

for cc in range(len(cuts)):
    M_ = np.random.randn(N,N)*g*scale
    uu,ss,vv = np.linalg.svd(M_)
    ss[cuts[cc]:] = 0
    M_ = uu @ np.diag(ss) @ vv
    sparse = np.random.rand(N,N)
    mask_J = np.random.rand(N,N)
    mask_J[sparse>p] = 0
    mask_J[sparse<=p] = 1
    M_ = np.zeros((N,N))#M_ * mask_J
    gain = .05  #for lower firing rate
    tau_r = 100*np.random.rand(N)  #important long self-spike time constant
    
    spks, rt, zt, M_ = G_FORCE_learning(M_, ft)
    MSEs_glm[cc] = sum((zt-ft)**2)/sum(zt**2)
    
# %%
plt.figure()
plt.plot(cuts, MSEs_glm, '-o')
plt.xlabel('initial rank',fontsize=40)
plt.ylabel('normalized MSE',fontsize=40)
plt.ylim([0, 8])

# %%
plt.figure()
plt.plot(ft)
plt.plot(zt[:])