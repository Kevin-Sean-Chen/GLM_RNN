# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:36:12 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
import itertools

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% Discrete-time binary neural network model
def spiking(p_spk, spk_past):
    """
    I: firing probability vector and past spike
    O: binary spiking vector
    """
    N = len(p_spk)
    rand = np.random.rand(N)
    spk = np.zeros(N)
    spk[p_spk>rand] = 1  # probablistic firing
    spk[spk_past==1] = 0  # refractoriness
    return spk

def nonlinearity(x):
    """
    I: current vector
    O: firing probability vector
    """
    base_rate = 0.0
    f = (1-base_rate)/(1+np.exp(-x)) + base_rate  # sigmoid nonlinearity
    return f

def current(Jij, spk):
    baseline = 0
    I = Jij @ spk + baseline  # coupling and baseline
    return I

# %% initializations
T = 10000
N = 5
gamma = 1.  # scaling coupling strength
Jij = np.random.randn(N,N)*gamma/N**0.5  # coupling matrix
sigma = np.zeros((N,T))

# %% Neural dynamics
for tt in range(0,T-1):
    p_spk = nonlinearity( current(Jij, sigma[:,tt]) )
    sigma[:,tt+1] = spiking(p_spk, sigma[:,tt])

plt.figure()
plt.imshow(sigma,aspect='auto',cmap='Greys',  interpolation='none')

# %% Inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% functions
eps = 10**-10
def inv_f(x):
    """
    Stable inverse function of sigmoid nonlinearity
    """
    invf = -np.log((1 / (x + eps)) - 1)  # stable inverse of sigmoid
    return invf

def J_inf(sigma):
    """
    Write into a scalable function once confirmed...
    """
    return Jij

# %% counting through time with the wild approxiamtion (ignoring other 'r' states)
Jinf = Jij*0  #inferred Jij

for ii in range(N):
    for jj in range(N):
        temp = sigma[[ii,jj],:]  # selectec i,j pair
        if ii is not jj:
            ### counting inference
            pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]  # find pair matching
            n10 = len(pos10)
            pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
            pos10_01 = np.intersect1d(pos10-1, pos01)  # find intersept one step behind
            n10_01 = len(pos10_01)
            p_1001 = n10_01 / n10  # normalize counts
            
            pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
            n00 = len(pos00)
            pos00_01 = np.intersect1d(pos00-1, pos01)
            n00_01 = len(pos00_01)
            p_0001 = n00_01 / n00
            
            Jinf[ii,jj] = inv_f(p_1001) - inv_f(p_0001)  # equation 11, ignoring r pattern
plt.figure()
mask = np.ones((N,N), dtype=bool)
np.fill_diagonal(mask, 0)
plt.plot(Jij[mask], Jinf[mask], 'ko')
plt.xlabel('true Jij', fontsize=30)
plt.ylabel('inferred Jij', fontsize=30)

# %% more careful calculaion of eqn. 11
n_comb = N-2  # the rest other than a pair
spins = [0,1]  # binary patterns
combinations = list(itertools.product(spins, repeat=n_comb))  # spin combinations
Jinf = Jij*0
neuron_index = np.arange(0,N)
dumy = 0
r_past = combinations[1]#[np.random.randint(N-2)]  # fix one past condition, can late use the frequent one

for ii in range(N):
    for jj in range(N):
        temp = sigma[[ii,jj],:]  # selectec i,j pair
        ind_r = [neuron_index[i] for i in range(N) if i!=ii and i!=jj]
        temp_r = sigma[ind_r,:]
        if ii is not jj:
            ### condition on patterns
            p_1001 = 0  # initialize for each pair
            p_0001 = 0
            pos_rpast =  np.where((temp_r.T == r_past).all(axis=1))[0]  # fixed conditional r pattern
            
            for rr in range(len(combinations)):
                r_ = combinations[rr]
                cond_r = np.where((temp_r.T == r_).all(axis=1))[0]  # all positions conditioned on pattern r
                
                pos10 = np.where((temp.T == (1,0)).all(axis=1))[0]
                pos10r = np.intersect1d(pos10, pos_rpast) #cond_r)
                n10r = len(pos10r)
                pos01 = np.where((temp.T == (0,1)).all(axis=1))[0]
                pos10_01 = np.intersect1d(pos10r-1, pos01)
                pos10_01r = np.intersect1d(pos10_01, cond_r-0)
                n10_01r = len(pos10_01r)
                if n10r is not 0:
                    p_1001r = n10_01r / n10r
                    p_1001 += p_1001r
                    
                pos00 = np.where((temp.T == (0,0)).all(axis=1))[0]
                pos00r = np.intersect1d(pos00, pos_rpast) #cond_r)
                n00r = len(pos00r)
                pos00_01 = np.intersect1d(pos00r-1, pos01)
                pos00_01r = np.intersect1d(pos00_01, cond_r-0)
                n00_01r = len(pos00_01r)
                if n00r is not 0:
                    p_0001r = n00_01r / n00r
                    p_0001 += p_0001r
                
#                if ii==1 and jj==2:  # checking counts
#                    dumy += n00_01
            
            Jinf[ii,jj] = inv_f(p_1001) - inv_f(p_0001)
plt.figure()
mask = np.ones((N,N), dtype=bool)
np.fill_diagonal(mask, 0)
plt.plot(Jij[mask], Jinf[mask], 'ko')
plt.xlabel('true Jij', fontsize=30)
plt.ylabel('inferred Jij', fontsize=30)

# %% Code for Max-Cal method
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% pseudo-code for now
Mxy = np.random.rand(N**2, N**2)  # all patterns to all patterns transition matrix
Mxy = Mxy / Mxy.sum(0)[:,None]
## delata example
gs = [(1,1,1), (2,2,1)]  #x,y,g(x,y)
beta = np.ones(len(gs))
Mxy_ = Mxy
for ii in range(len(gs)):
    g = gs[ii]
    Mxy_[g[0],g[1]] = Mxy_[g[0],g[1]] * np.exp(beta[ii]*g[2])  # exp tilted transition matrix

# %%
uu, vr = np.linalg.eig(Mxy_)  # right vector
vl = np.linalg.inv(vr)  # left vector
lamb = uu[0]  # larget eig
Pyx = vr/(lamb*(vl))*Mxy_  # posterior.... not  sure about this?
def lamb_beta(beta):
    # return largest eigen value given beta vector
    lamb=None
    return lamb

def expect_g(beta):
    # put in posterior calculation and stationary probability
    g_bar=None
    return g_bar

def objective(beta):
    g_bar = expect_g(beta)
    obj = np.dot(beta, g_bar) - lamb_beta(beta)
    return -obj # do scipy.minimization on this
