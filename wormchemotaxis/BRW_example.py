#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:23:22 2022

@author: kschen
"""
import numpy as np
from matplotlib import pyplot as plt

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30)

# %% 1-D random walk
lt = 1000 # lenth of simulation
N = 100 # number of particles
sig = 0.5 # noise strength
xs_rw = np.zeros((N, lt))

for tt in range(lt-1):
    xs_rw[:,tt+1] = xs_rw[:,tt] + np.random.randn(N)*sig

plt.figure()
plt.plot(xs_rw.T)

# %% biased random walk
p = 0.55 # biased term
bias = np.ones(N)  # a simple way to control the sign
xs_brw = np.zeros((N, lt))

for tt in range(lt-1):
    biast = bias.copy()
    rand = np.random.rand(N)
    biast[np.where(rand>p)[0]] = -1
    xs_brw[:,tt+1] = xs_brw[:,tt] + biast*np.random.rand(N)*sig

plt.figure()
plt.plot(xs_brw.T)

# %% pdf of the tracks
plt.figure()
plt.hist(xs_rw[:,-100:].flatten(),20)
plt.hist(xs_brw[:,-100:].flatten(),20,alpha=0.8)
plt.legend({'random walk','biased-random walk'})

# %% correlated biased-random walk
tau = 200 # correlation time-scale
dt = 0.1 # discrete time steps
xs_crw = np.zeros((N, lt))

for tt in range(lt-1):
    biast = bias.copy()
    rand = np.random.rand(N)
    biast[np.where(rand>p)[0]] = -1
    xs_crw[:,tt+1] = xs_crw[:,tt] + dt/tau*(-xs_crw[:,tt]) + 1/tau*np.sqrt(dt)*biast*np.random.rand(N)

plt.figure()
plt.plot(xs_crw.T)
# %% measure correlation
def autocorr(x):
    mean = np.mean(x)
    var = np.var(x)
    nx = x - mean
    acorr = np.correlate(nx, nx, 'full')[len(nx)-1:] 
    acorr = acorr / var / len(nx)
    return acorr

acf_rand = autocorr(np.random.randn(lt))
acf_rw = autocorr(xs_rw[1,:])
acf_crw = autocorr(xs_crw[1,:])
plt.figure()
plt.plot(acf_rand)
plt.plot(acf_rw)
plt.plot(acf_crw)
plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$\langle x(t)x(t-\Delta t)\rangle$')
plt.legend({'random','random walk','biased-random walk'})

# %% energy-landscape driven
def FEP(x):
    return -x

gamma = 100
xs_fep = np.zeros((N, lt))
vs_fep = xs_fep*0

for tt in range(lt-1):
    biast = bias.copy()
    rand = np.random.rand(N)
    biast[np.where(rand>p)[0]] = -1
    xs_fep[:,tt+1] = xs_fep[:,tt] + dt*vs_fep[:,tt]
    vs_fep[:,tt+1] = vs_fep[:,tt] + dt/tau*(-gamma*vs_fep[:,tt] + FEP(xs_fep[:,tt])) + np.sqrt(dt)*biast*np.random.rand(N)

plt.figure()
plt.plot(xs_fep.T)

# %% PCA with tracks, since we talked about PCA!
C = np.cov(xs_fep.T)
uu,ss,vv = np.linalg.svd(C)
plt.figure()
plt.plot(uu[:,:3])