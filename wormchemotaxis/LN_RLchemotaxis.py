# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:36:08 2020

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

# %% functions
def Environement(xy,C0,sig,target):
    """
    Given location in 2D x,y and the scale factor C0, width sig, and target location, return environment concentration E
    """
    x,y = xy
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

def Q_val(S,A,k,h):
    """
    A test for Q value function that takes a kernel operation on state S and past action A
    """
    q = 1/(1+np.exp(np.dot(k @ Ks,S) + np.dot(h @ Ks,A)))
    return q
   
def dist(xy,targ):
    """
    Eucledian distance between two 2D points
    """
    return np.sqrt(np.sum((xy-targ)**2))

def sensing_step(Et,St,At,tt,xy):
    """
    Return 
    """
    Et[tt] = Environement(xy,C0,sig,target)  #reward proxy (goal is indeed to maximize this...)
    Et_ = Et[tt-kl:tt]
#    R_ = Et[-1]
    St[tt] = Sensory(Et_,K)  #state representation
    St_ = St[tt]
    At[tt] = Action(St_, N)  #action in space
    At_ = At[tt-kl:tt]
    return Et_, At_, Et,St,At

def kinematic_step(xy,dth,A):
    """
    Return next location and angle given current location, angle, and action
    """
    x,y = xy
    dth = dth + np.random.randn()*thr + A*(np.random.rand()*2*np.pi-np.pi)
    dd = (v+vr*np.random.randn())*dt
    x = x + dd*np.sin(dth)
    y = y + dd*np.cos(dth)
    xy = x,y
    return xy, dth

# %% chemotaxis setttings
trials = 100
dt = 0.1
T = 1000
time = np.arange(0,T,dt)
tl = len(time)
kl = 50  #kenrel length
nb = 5  #number of basis for the kenel
Ks = (np.fliplr(basis_function1(kl,nb).T).T).T  #basis function
#K = np.dot(np.random.randn(nb), Ks)  #constructing the kernel with basis function
N = .01  #scaling of logistic nonlinearity
thr = 0.05  #noise strength on angle change
v = 1  #mean velocity
vr = 0.1  #noise strength on velocity
C0 = 200  #concentration scaling
sig = 100  #width of concentration profile
target = np.array([10,10])  #traget position
eps = 0.5  #tolarated distance from the target

# %% semi-gradient SRSA
alpha = 0.1
gamma = 0.5
k = np.random.randn(nb)  #state kernel
h = np.random.randn(nb)  #action kernel
for rr in range(trials):
    #initial state and action
    xy = np.random.randn(2)
    dth = np.random.randn()*2*np.pi-np.pi
    Et, St, At = np.zeros(tl), np.zeros(tl), np.zeros(tl)
    for tt in range(kl,tl):
        ### update chemotaxis measurements
        Et_, At_, Et,St,At = sensing_step(Et,St,At,tt,xy)
        R_ = Et_[-1]
        ### update kinematics
        xy_, dth_ = kinematic_step(xy,dth, At_[-1])
        ### terminal update
        if dist(xy_,target)<eps or tt>tl:
            k = k + alpha*(R_-Q_val(Et_,At_,k,h))* (Ks @ Et_)*Q_val(Et_,At_,k,h)*(1-Q_val(Et_,At_,k,h))
            h = h + alpha*(R_-Q_val(Et_,At_,k,h))* (Ks @ At_)*Q_val(Et_,At_,k,h)*(1-Q_val(Et_,At_,k,h))
            break
        
        Et_next, At_next, _,_,_ = sensing_step(Et,St,At,tt,xy_)
        k = k + alpha*(R_+gamma*Q_val(Et_next,At_next,k,h)-Q_val(Et_,At_,k,h))* (Ks @ Et_)*Q_val(Et_,At_,k,h)*(1-Q_val(Et_,At_,k,h))
        h = h + alpha*(R_+gamma*Q_val(Et_next,At_next,k,h)-Q_val(Et_,At_,k,h))* (Ks @ At_)*Q_val(Et_,At_,k,h)*(1-Q_val(Et_,At_,k,h))
        
#        print(k)
#        ###evaluate next step
#        #tumble
#        xy_t, dth_t = kinematic_step(xy,dth,1) #At_[-1]
#        Et_t, At_t, Ett,Stt,Att = sensing_step(Et,St,At,tt,xy_t)
#        #run
#        xy_r, dth_r = kinematic_step(xy,dth,0)
#        Et_r, At_r, Ert,Srt,Art = sensing_step(Et,St,At,tt,xy_r)
#        ### loop update
#        if Q_val(Et_t,At_t,k,h)>=Q_val(Et_r,At_r,k,h):
#            print(1)
#            xy, dth = xy_t, dth_t
#            k = k + alpha*(R_+gamma*Q_val(Et_t,At_t,k,h)-Q_val(Et_,At_,k,h))* (Ks @ Et_)
#            h = h + alpha*(R_+gamma*Q_val(Et_t,At_t,k,h)-Q_val(Et_,At_,k,h))* (Ks @ At_)
#            Et,St,At = Ett,Stt,Att
#        elif Q_val(Et_t,At_t,k,h)<Q_val(Et_r,At_r,k,h):
#            print(0)
#            xy, dth = xy_r, dth_r
#            k = k + alpha*(R_+gamma*Q_val(Et_r,At_r,k,h)-Q_val(Et_,At_,k,h))* (Ks @ Et_)
#            h = h + alpha*(R_+gamma*Q_val(Et_r,At_r,k,h)-Q_val(Et_,At_,k,h))* (Ks @ At_)
#            Et,St,At = Ert,Srt,Art
    print(k)
# %%
plt.figure()
plt.plot(k @ Ks,label='state K')
plt.plot(h @ Ks,'--',label='action h')
plt.legend()
