# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:41:55 2022

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.linalg import hankel

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% fixed white noise
###############################################################################
# %% parameters
K = 10.
tau = 10.
base_p = 0.1

T = 1000
dt = 0.1
time = np.arange(0,T,dt)
lt = len(time)

# %% dynamics
Tt = np.zeros(lt)
Ct = np.random.randn(lt)*0.1
St = np.zeros(lt)
for tt in range(1,lt):
    Tt[tt] = Tt[tt-1] + dt*1/tau*( -Tt[tt-1] + K*(1-np.exp(-Ct[tt-1]/K)))
    diff = Ct[tt] - Tt[tt]
    if diff<=0:
        St[tt] = np.random.poisson(-diff*dt + base_p*dt)

# %% response
D = 50+1
X = sp.linalg.hankel(np.append(np.zeros(D-2),Ct[:lt-D+2]),Ct[lt-D+1:])
K_mle = np.linalg.pinv(X.T @ X) @ X.T @ St
plt.figure()
plt.plot(K_mle)

# %%
###############################################################################
# %% parameters for environment and behavior
def gradient(C0, targ, dDT, x, y):
    concentration = C0/(4*np.pi*dDT)*np.exp(-(x-targ)**2/(4*dDT))  #depends on diffusion conditions along x-axis
    #concentration = np.max((0,C0*x))
    return concentration

pert = np.random.randn(lt)*.0
C0 = 1000
dDT = 50
targ = 10
v = .5
kappa = 2

# %%
xyt = np.zeros((2,lt)) #np.random.randn(2,lt) #
xyt[:,1] = np.random.randn(2)
tht = np.zeros(lt)
Tt = np.zeros(lt)
Ct = np.zeros(lt)#np.random.randn(lt)
St = np.zeros(lt)

for tt in range(1,lt):
    Ct[tt] = gradient(C0, targ, dDT, xyt[0,tt-1], xyt[1,tt-1]) + pert[tt]
    Tt[tt] = Tt[tt-1] + dt*1/tau*( -Tt[tt-1] + K*(1-np.exp(-Ct[tt]/K)))
    diff = Ct[tt] - Tt[tt]
    if diff<=0:
        St[tt] = np.random.poisson(-diff*dt)
    elif base_p>np.random.rand():
        St[tt] = 0#np.random.poisson(base_p*dt)
    if St[tt]>0:
        dth = (np.random.randn()*np.pi*2-np.pi)/dt
        #np.random.rand()*np.pi*2-np.pi
    else:
        dth = np.random.randn()*kappa
    
    tht[tt] = tht[tt-1] + dth*dt
    veff = v * np.array([np.cos(tht[tt]), np.sin(tht[tt])])
    xyt[:,tt] = xyt[:,tt-1] + veff*dt
    
# %%
plt.figure()
plt.plot(xyt[0,:], xyt[1,:])
plt.figure()
plt.plot(St)
plt.plot(Ct)

# %% response
D = 50+1
X = sp.linalg.hankel(np.append(np.zeros(D-2),Ct[:lt-D+2]),Ct[lt-D+1:])
K_mle = np.linalg.pinv(X.T @ X) @ X.T @ St
plt.figure()
plt.plot(K_mle)
