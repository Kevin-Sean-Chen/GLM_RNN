# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 08:20:39 2022

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

###
# Aim to simulate latent dynamcis
# generate spike trains
# fit with GLM network
# ... extentions:
#               adding hidden units
#               recovery with latent (GP or LDS) or network (rank, sparsity, EI) constraints
#               test with perturbation (out-of-equilibrium behavior)
###
# %% network and time length settings
N = 10
T = 100
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)
sig = .5
tau_l = 5

# %% latent dynamics
# potential: V(x) = 1/4*x**4 - x**2 - c*x
c = 0.
def vf(x):
    return -(x**3 - 2*x - c)
latent = np.zeros(lt)
for tt in range(lt-1):
    latent[tt+1] = latent[tt] + dt/tau_l*(vf(latent[tt])) + np.sqrt(dt*sig)*np.random.randn()

latent = np.sin(time/20)

plt.figure()
plt.plot(time,latent)

# %% spiking process
M = np.random.randn(N)*.5  # N by latent D
M = np.ones(N)*1.5
b = 3  # offiset for LDS
J = np.random.randn(N,N)*10.
randM = np.random.randn(N,N)
rank = 2
UU,SS,VV = np.linalg.svd(randM)
v1, v2 = UU[:,:rank], VV[:rank,:]
J = (v1 @ v1.T + v2.T @ v2)*30 + J*0 + 0*v1@v2

spk = np.zeros((N,lt))  # spike train
rt = spk*1  # spike rate
tau_r = np.random.rand(N)*5
lamb_max, lamb_min = 1, 0
def NL(x):
    """
    Spiking nonlinearity
    """
#    nl = x * (x > 0)
    nl = (lamb_max-lamb_min)/(1+np.exp(-x)) + lamb_min
#    nl = np.log(1+np.exp(x))
#    nl = np.exp(x)
    return nl
def phi(x):
    """
    Synaptic nonlinearity
    """
#    ph = lamb_max/(1+np.exp(-x)) + lamb_min
#    ph = np.tanh(x)
    ph = x
    return ph
    

for tt in range(lt-1):
    spk[:,tt] = np.random.poisson(NL(M*latent[tt]-b))
#    spk[:,tt] = np.random.poisson(NL(J @ phi(rt[:,tt]) + 0)*dt)  # matched model control
    rt[:,tt+1] = rt[:,tt] + dt/tau_r*(-rt[:,tt] + spk[:,tt])

plt.figure()
plt.imshow(rt,aspect='auto')#,cmap='gray')

# %% inference
def negLL(ww, spk, rt, dt, f=np.exp, lamb=0):
    N = spk.shape[0]
    b = ww[:N]
    W = ww[N:].reshape(N,N)
    # evaluate log likelihood and gradient
    ll = np.sum(spk * np.log(f(W @ phi(rt) + b[:,None])) - f(W @ phi(rt) + b[:,None])*dt) \
            - lamb*np.linalg.norm(W)
#            - lamb*(W.T @ W).sum()
    return -ll

dd = N*N+N
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, spk,rt,dt,NL, 10.),w_init,method='L-BFGS-B',tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

###
### try row wise iteration for independent regression
### then try bilinear temporal filter
### then add in self-history inhibition

# %%
brec = w_map[:N]
Wrec = w_map[N:].reshape(N,N)*1.
spk_rec = np.zeros((N,lt))
rt_rec = spk_rec*0
for tt in range(lt-1):
    spk_rec[:,tt] = np.random.poisson(NL(Wrec @ phi(rt_rec[:,tt]) + brec)*dt)
    rt_rec[:,tt+1] = rt_rec[:,tt] + dt/tau_r*(-rt_rec[:,tt] + spk_rec[:,tt]) 

plt.figure()
plt.imshow(rt_rec,aspect='auto')

# %% rank analysis
c_rt = np.cov(rt)
c_rec = np.cov(rt_rec)

plt.figure()
uu,ss,vv = np.linalg.svd(c_rt)
plt.subplot(221)
plt.plot(ss,'-o')
plt.title('activity rank',fontsize=30)
uu,ss,vv = np.linalg.svd(c_rec)
plt.subplot(223)
plt.plot(ss,'-o')
uu,ss,vv = np.linalg.svd(J)
plt.subplot(222)
plt.plot(ss,'k-o')
plt.title('network rank',fontsize=30)
uu,ss,vv = np.linalg.svd(Wrec)
plt.subplot(224)
plt.plot(ss,'k-o')

# %% compare latent and rank projection
uu,ss,vv = np.linalg.svd(np.cov(rt))
m_pc = vv[0,:] #uu[:,0]
uu,ss,vv = np.linalg.svd(Wrec)
m_con = vv[0,:] #uu[:,0]
plt.figure()
plt.plot(latent, label='latent')
plt.plot(m_pc/np.linalg.norm(m_pc) @ rt, label='activity')
plt.plot(m_con/np.linalg.norm(m_con) @ rt_rec, label='structure')
plt.legend(fontsize=30)
