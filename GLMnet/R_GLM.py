# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:45:15 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import math
from scipy.optimize import minimize
from scipy.special import gamma, factorial

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# test out R-GLM model

# %%
def NL(x):
#    nl = np.log(1+np.exp(x))
    nl = 50/(1+np.exp(-x))
    return nl

def spiking(nl):
    spk = np.random.poisson(nl)
    return spk

# %% intiializ parameters
N = 10
ld = 2
A = np.random.randn(ld,ld)*.1
A = np.array([[0.95,-.1],[.1,0.8]])*1.
thet = np.pi/100
A = np.array([[np.cos(thet), np.sin(thet)], [-np.sin(thet), np.cos(thet)]])*.9

W = np.random.randn(ld,N)*.1
C = np.random.randn(N,ld)*.5
Z = np.tril(np.random.randn(N, N)*0.1, k=-1)
#Z = np.random.randn(N,N)
b = np.random.randn(ld)*0.1
d = np.random.randn(N)*0.1

# %%
# time series
dt = 0.1
T = 200
tau = 2
time = np.arange(0,T,dt)
lt = len(time)
x = np.random.randn(ld,lt)
y = np.zeros((N,lt))
rt = np.zeros((N,lt))

for tt in range(lt-1):
    x[:,tt+1] = A@x[:,tt] + W@y[:,tt] + b + np.random.randn(ld)*.1
    y[:,tt+1] = spiking( NL(C@x[:,tt]+Z@y[:,tt]+d)*dt )
    rt[:,tt+1] = rt[:,tt] + dt/tau*(-rt[:,tt] + y[:,tt])

plt.figure()
plt.imshow(y, aspect='auto')

# %%
def ww2mat(ww):
    A = ww[:ld**2].reshape(ld,ld)
    C = ww[ld**2:ld**2+N*ld].reshape(N,ld)
    zz = ww[ld**2+N*ld: ld**2+N*ld+(N*(N-1)//2)]
    Z = np.zeros((N,N))
    Z[np.tril_indices(N,k=-1)] = zz
    W = ww[ld**2+N*ld+(N*(N-1)//2): ld**2+N*ld+(N*(N-1)//2)+N*ld].reshape(ld,N)
    b = ww[ld**2+N*ld+(N*(N-1)//2)+N*ld: ld**2+N*ld+(N*(N-1)//2)+N*ld+ld]
    d = ww[-N:]
    return A,C,Z,W,b,d

def nll_rglm(ww, spk, xx, dt, f, lamb=0):
    A,C,Z,W,b,d = ww2mat(ww)
    # poisson log likelihood
    ###
    # use sparse matrix to construct latent x!
    ###
    temp = C@xx + Z@spk + d[:,None]
    ll = np.sum(spk * np.log(f(temp)) - f(temp)*dt) \
            - lamb*np.linalg.norm(A-np.eye(ld)) \
            - lamb*np.sum((xx[:,1:] - A@xx[:,:-1] - W@spk[:,:-1] - b[:,None])**2)        
    return -ll

dd = ld**2+N*ld+(N*(N-1)//2)+N*ld+ld+N
w_init = np.zeros([dd,])+0.1  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: nll_rglm(w, y,x,dt,NL, .51),w_init,method='L-BFGS-B')#,tol=1e-5)
w_map = res.x
print(res.fun)
print(res.success)

# %%
