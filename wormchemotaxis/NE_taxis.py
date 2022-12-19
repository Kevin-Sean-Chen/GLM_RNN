# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 01:41:31 2022

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy import optimize

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %%
def KL(P,Q):
    epsilon = 10**-20
    P = P+epsilon
    Q = Q+epsilon
    divergence = np.sum(P*np.log(P/Q))
    return divergence

def control(pars):
    mu, var = pars
    pq = -0.5*(xx-mu)**2/var  ### 2D gaussian for x and v control
    pq = pq/np.sum(pq)
    return pq

def potential(pars, parE):
    mu, var = pars
    muE, varE = parE
    E = -0.5*(mu-muE)**2/var  #for can change!
    return E.sum()

def free_energy(pars, T, parE, par0):
    P = control(pars)
    Q = control(par0)
    FE = potential(pars, parE) - T*KL(P,Q)
    return -FE   
    
# %%
xx = np.arange(-10,10,0.1)
Ts = np.arange(0.1,2.1,0.2)
parE = 5,10
par0 = 0,5
par_opt = []
fs = []
for ii in range(len(Ts)):
    x0 = np.random.randn()*1, np.random.rand()
    res = optimize.minimize(free_energy, x0, args=(Ts[ii], parE, par0), \
                            bounds = ((-10, 10), (0, 50)),\
                            options={'disp': True}, tol=1e-5)#, method = "BFGS")
    par_opt.append(res.x)
    fs.append(-free_energy(res.x,  Ts[ii], parE, par0))
    
# %%
plt.figure()
plt.plot(1/Ts, np.array(par_opt)[:,0],'-o')
plt.plot(1/Ts, 10*np.array(fs), '-o')

# %% navigation with each step as dF/dp

# %%
###############################################################################
# %%
tau1 = 2/3
tau2 = 2/3
g1 = 1
g2 = 1
w0 = 1
wfb = 1 #0,-1
w21 = 1
k1 = 15
k2 = 5
k3 = 15
b1 = 0
b2 = 1.5
b22 = 1.5
gam1 = 1
gam2 = 1
c1 = 0.5
c2 = 0

tau0 = 1
g0 = 1
vl0,vl1,vl2 = 0.5,0.5,0.1
R = 1
vv = 1
# %%
def CT(x):
    return 1*x  #for linear gradient along x
T = 1000
dt = 0.1
Vs = np.zeros((3,T))
xt = np.zeros(T)
Mt = np.zeros(T)
xyt = np.zeros((2,T))
thet = np.zeros(T)
for tt in range(1,T-1):
    # synaptic input
    F10 = w0*Vs[0,tt]
    F12 = wfb*(1/(1+np.exp(-k1*(Vs[2,tt] - b1))))
    F21 = w21*(gam1/(1+np.exp(-k2*(Vs[1,tt]-b2)))) - gam2/(1+np.exp(-k3*(Vs[1,tt]-b22)))
    I = CT(xt[tt]) - CT(xt[tt-1])
    # dynamics
    Vs[0,tt+1] = Vs[0,tt] + dt/tau0*(-g0*(Vs[0,tt] - vl0) + I*R)
    Vs[1,tt+1] = Vs[1,tt] + dt/tau1*(-g1*(Vs[1,tt] - vl1) + F10*Vs[0,tt] + F12*Vs[2,tt])
    Vs[2,tt+1] = Vs[2,tt] + dt/tau2*(-g2*(Vs[2,tt] - vl2) + F21*Vs[1,tt]) 
    # behavioral
    if Vs[2,tt]>0:
        Mt[tt] = 1
    elif Vs[2,tt]<=0:
        Mt[tt] = 0
    # random walk
    dxy = np.array([np.cos(thet[tt-1]), np.sin(thet[tt-1])])*vv
    xyt[:,tt+1] = xyt[:,tt] + dt*dxy
    xt[tt] = xyt[0,tt]
    if Mt[tt]==0:
        thet[tt] = -thet[tt-1]
    else:
        thet[tt] = thet[tt-1] + np.random.randn()
# %%
plt.figure()
plt.plot(xyt[0,:], xyt[1,:])