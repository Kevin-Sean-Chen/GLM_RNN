# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:41:01 2021

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

# %% parameters
### neurons
N = 300 #number of neuron
dt = 0.00005
tref = 0.002 #refractory time constanct in s
tm = 0.01  #membrane time constant
vreset = -65  #respet voltage
vpeak = -40  #spiking threshold
td = 0.02  #synaptic decay
tr = 0.002  #synaptic rise

### network
alpha = dt*0.1  #rate of change
Pinv = np.eye(N)*alpha  #init for IRLS
p = 0.1  #sparsity

### target dynamics
T = 15
imin = 1#round(5/dt)
icrit = len(ft)#round(10/dt)
step = 2#50
nt = len(ft)#round(T/dt)
Q = 10
G = 0.04
zx = .01*ft.copy()#np.sin(2*np.pi*dt*50*np.arange(0,nt))

#T = 15
#imin = round(5/dt)
#icrit = round(10/dt)
#step = 50
#nt = round(T/dt)
#Q = 10
#G = 0.04
#zx = np.sin(2*np.pi*dt*5*np.arange(0,nt))

# %% initialization
### recordings
k = 1#min(zx.shape)
IPSC = np.zeros(N)  #post-synaptic current
h = np.zeros(N)  #filtered spike
r = np.zeros(N)  #filtered rate
hr = np.zeros(N)  #filtered hr
JD = 0*IPSC  #spike time
tspike = np.zeros((4*nt,2)) #each spike time
ns = 0  #number of spikes
z = 0#np.zeros(k)  #approximation

### initial activity
v = vreset + np.random.randn(N)*(30-vreset)  #initial voltage
v_ = v
RECB = np.zeros((nt,10))  #synaptic weight
mask = np.random.rand(N,N)
mask[mask>p] = 0
OMEGA = G*np.random.randn(N,N)*mask/(np.sqrt(N)*p)  #initial weight matrix
OMEGA = OMEGA*mask
BPhi = np.zeros(N)  #initial learning matrix

### Connectivity matrix
for ii in range(N):
    QS = np.where(np.abs(OMEGA[ii,:])>0)[0]
    OMEGA[ii,QS] = OMEGA[ii,QS] - sum(OMEGA[ii,QS])/len(QS)
E = (2*np.random.rand(N)-1)*Q
REC2 = np.zeros((nt,20))
REC = np.zeros((nt,10))
current = np.zeros(nt)  #record output current
i = 1

# %% Dynamics
tlast = np.zeros(N)  #refractory time
BIAS = vpeak#.copy()  #bias current
ilast = i
for ii in range(ilast,nt):
    I = IPSC + E*z + BIAS #neural current
    dv = (dt*ii>tlast+tref)*(-v+I)/tm  #voltage with refractory period
    v = v + dt*dv
    index = np.where(v>=vpeak)[0]  #find spiking neuron
    
    if len(index)>0:
        JD = np.sum(OMEGA[:,index],1)  #current due to spike
        tspike[ns:ns+len(index),:] = np.concatenate((index[:,None], (0*index+dt*ii)[:,None]),axis=1)
        ns = ns + len(index)  #number of spikes
    tlast = tlast + (dt*ii- tlast)*(v>=vpeak)
        
    if tr==0:
        IPSC = IPSC*np.exp(-dt/td) + JD*(len(index)>0)/td
        r = r*np.exp(-dt/td) + (v>=vpeak)/td
    else:
        IPSC = IPSC*np.exp(-dt/tr) + h*dt
    
        h = h*np.exp(-dt/td) + JD*(len(index)>0)/(tr*td)
        r = r*np.exp(-dt/tr) + hr*dt
        hr = hr*np.exp(-dt/td) + (v>=vpeak)/(tr*td)
    
    ### IRLS
    z = np.dot(BPhi, r)
    err = z - zx[ii]
    if np.mod(ii,step)==1:
        if ii>imin:
            if ii<icrit:
                cd = Pinv @ r
#                BPhi = BPhi - (cd*err)
                Pinv = Pinv - (np.outer(cd,cd))/(1+np.dot(r,cd))
                
    v = v + (30-v)*(v>=vpeak)
    REC[ii,:] = v[:10]
    v = v+(vreset-v)*(v>=vpeak)
    current[ii] = z
    RECB[ii,:] = BPhi[:10]
    REC2[ii,:] = r[:20]
    
# %% analysis
plt.figure()
plt.plot(zx)
plt.plot(current,'--')
plt.title('spk-FORCE;M=1000',fontsize=40)
