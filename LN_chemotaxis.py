# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:24:11 2020

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
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5
# %% Simple LN chamotaxis
def Environement(x,y,C0,sig,target):
    """
    Given location in 2D x,y and the scale factor C0, width sig, and target location, return environment concentration E
    """
    E = C0*np.exp(-((x-target[0])**2+(y-target[1])**2)/sig)
    return E

def Sensory(E,K):
    """
    Given envioronment concentration E and the kernel K return sensory activation S
    """
    S = np.dot(E,K)
    return S

def Action(S,N):
    """
    Given the sensory activation S and nonlinearity N return an action A
    """
    P = 1/(1+np.exp(N*S))
    if np.random.rand()<P:
        A = 1
    else:
        A = 0
    return A

def SimTask(Ns, pars, time):
    """
    Given N simulations, parameters for chemotaxis pars, and time vector, return N of three time series Et, St, At
    """
    C0, sig, target, K, N, thr, v, vr = pars
    dt = time[1]-time[0]
    tl = len(time)
    kl = len(K)
    Et, St, At = np.zeros((Ns,tl)), np.zeros((Ns,tl)), np.zeros((Ns,tl))
    ###iteration through repeats
    for nn in range(Ns):
        x,y = np.random.randn(2)
        dth = np.random.randn()*2*np.pi-np.pi
        ###iteration through time
        for tt in range(kl,tl):
            ### update chemotaxis measurements
            Et[nn,tt] = Environement(x,y,C0,sig,target)
            Et_ = Et[nn,tt-kl:tt]
            St[nn,tt] = Sensory(Et_,K)
            St_ = St[nn,tt]
            At[nn,tt] = Action(St_, N)
            At_ = At[nn,tt]
            ### update kinematics
            dth = dth + np.random.randn()*thr + At_*(np.random.rand()*2*np.pi-np.pi)
            dd = (v+vr*np.random.randn())*dt
            x = x + dd*np.sin(dth)
            y = y + dd*np.cos(dth)
    return Et, St, At

def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    return np.array(BBstm).T

# %% parameters and simulation
dt = 0.1
T = 1000
time = np.arange(0,T,dt)
tl = len(time)
thr = 0.05  #noise strength on angle change
v = 1  #mean velocity
vr = 0.1  #noise strength on velocity
C0 = 200  #concentration scaling
sig = 100  #width of concentration profile
target = np.array([10,10])  #traget position
kl = 50  #kenrel length
nb = 5  #number of basis for the kenel
K = np.dot(np.random.randn(nb), (np.fliplr(basis_function1(kl,nb).T).T).T)  #constructing the kernel with basis function
N = .1  #scaling of logistic nonlinearity
Ns = 10  #number of repetitionsd
pars = C0, sig, target, K, N, thr, v, vr
Et, St, At = SimTask(Ns, pars, time)

plt.plot(Et.T)

# %% performance scan

# %% information analysis