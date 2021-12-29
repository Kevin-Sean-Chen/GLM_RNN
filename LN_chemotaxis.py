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

def Action(S,N,maxp,basep):
    """
    Given the sensory activation S and nonlinearity N return an action A
    with maxp and basep controlling the maximum and baselone turning rate
    """
    P = maxp/(1+np.exp(-N*S)) + basep
    if np.random.rand()<P:
        A = 1
    else:
        A = 0
    return A

def SimTask(Ns, pars, time):
    """
    Given N simulations, parameters for chemotaxis pars, and time vector, return N of three time series Et, St, At
    """
    C0, sig, target, K, N, thr, v, vr, maxp, basep, E_n = pars
    dt = time[1]-time[0]
    tl = len(time)
    kl = len(K)
    Et, St, At = np.zeros((Ns,tl)), np.zeros((Ns,tl)), np.zeros((Ns,tl))
    xs,ys,dths = np.zeros((Ns,tl)), np.zeros((Ns,tl)), np.zeros((Ns,tl))
    ###iteration through repeats
    for nn in range(Ns):
        x,y = np.random.randn(2)
        dth = np.random.randn()*2*np.pi-np.pi
        ###iteration through time
        for tt in range(kl,tl):
            ### update chemotaxis measurements
            Et[nn,tt] = Environement(x,y,C0,sig,target) + E_n*np.random.randn()
            Et_ = Et[nn,tt-kl:tt]
            St[nn,tt] = Sensory(Et_,K)
            St_ = St[nn,tt]
            At[nn,tt] = Action(St_, N, maxp, basep)
            At_ = At[nn,tt]
            ### update kinematics
            delta_th = At_*(np.random.rand()*2*np.pi-np.pi)
            dth = dth + np.random.randn()*thr + delta_th
            dd = (v+vr*np.random.randn())*dt
            x = x + dd*np.sin(dth)
            y = y + dd*np.cos(dth)
            
            xs[nn,tt], ys[nn,tt], dths[nn,tt] = x,y,delta_th
            
    return Et, St, At,  xs,ys,dths

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
thr = 0.01  #noise strength on angle change
v = 1.0  #mean velocity
vr = 0.1  #noise strength on velocity
C0 = 100  #concentration scaling
sig = 60  #width of concentration profile
E_n = 0  #environemental noise
target = np.array([20,20])  #traget position
kl = 50  #kenrel length
nb = 5  #number of basis for the kenel
#K = np.dot(np.random.randn(nb), (np.fliplr(basis_function1(kl,nb).T).T).T)  #constructing the kernel with basis function
N = 10.  #scaling of logistic nonlinearity
Ns = 30  #number of repetitionsd
maxp = 0.7 #maximum turning probability
basep = 0.2  #baseline turning (exploration)
pars = C0, sig, target, K, N, thr, v, vr, maxp, basep, E_n
Et, St, At, xs,ys,dths = SimTask(Ns, pars, time)

plt.figure()
plt.subplot(311)
plt.plot(time,Et.T)
plt.ylabel('E_t')
plt.subplot(312)
plt.plot(time,St.T)
plt.ylabel('S_t')
plt.subplot(313)
plt.plot(time,At.T)
plt.ylabel('A_t')
plt.xlabel('time')

plt.figure()
plt.plot(xs.T, ys.T)
# %% performance scan
###############################################################################
# %% information analysis
from mutual_info import mutual_information_2d
IES = np.zeros(Ns)
ISA = np.zeros(Ns)
IEA = np.zeros(Ns)
for nn in range(Ns):
    IES[nn] = mutual_information_2d(Et[nn,:], St[nn,:], sigma=1, normalized=False)
    ISA[nn] = mutual_information_2d(St[nn,:], At[nn,:], sigma=1, normalized=False)
    IEA[nn] = mutual_information_2d(Et[nn,:], At[nn,:], sigma=1, normalized=False)
plt.figure()
ind = np.arange(3)
plt.plot(ind,[IES, ISA, IEA])
plt.xticks(ind, ('I(E,S)', 'I(S,A)', 'I(E,A)'))
plt.ylabel('bits')

# %% scanning
slopes = np.array([0.01,0.1,1,10,100,1000])  #nonlinearity
rec = np.zeros((len(slopes),Ns))
for ss in range(len(slopes)):
    pars = C0, sig, target, K, slopes[ss], thr, v, vr, maxp, basep, E_n
    Et, St, At, xs,ys,dths = SimTask(Ns, pars, time)
    for nn in range(Ns):
        rec[ss,nn] = mutual_information_2d(Et[nn,:], dths[nn,:], sigma=1, normalized=False)

# %%
plt.figure()
plt.semilogx(slopes,rec,'.')
plt.semilogx(slopes,np.mean(rec,axis=1),'-o')
plt.xlabel('nonlinearity')
plt.ylabel('I(E,A)')
