# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 20:49:39 2022

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% biophysical functions
def synaptic_update(s, V, syn_params):
    ar,beta,V_th,ad = syn_params
    ds = ar/(1+np.exp(-beta*(V-V_th)))*(1-s) - ad*s
    return ds

def voltage_update(V, s, volt_params, syn_params, Istim):
    C, g, Js, Jg, Ec, Er = volt_params
    vi,vj = np.meshgrid(V, V)
    Ig = (Jg*(vi-vj)).sum(0)
    Is = Js @ synaptic_update(s, V, syn_params)*(V-Er)
    dv = -g*(V-Ec) - Ig - Is + Istim*inpt
    return dv

# %% biphysical parameters
ar = 1
ad = 5
beta = 0.125
Ec = -35
Er = -45
g = 10
C = 1
Js = np.array([[10, -3, 5], [-3, 20, 0], [-10, 0, 0]])*5
Jg = np.zeros((3,3))*0
Jg[1,2], Jg[2,1] = 1,1
V_th = -30
syn_params = ar, beta, V_th, ad
volt_params = C, g, Js, Jg, Ec, Er
N = 3
inpt = np.array([1,0,0])
noise = 10

# %% time and stimulu
dt = 0.01  #dt
T = 100  #10s
time = np.arange(0,T,dt)
lt = len(time)
Istim = np.zeros(lt)
Istim[2000:2500] = 100

# %% initialize
Vt = np.zeros((N,lt))
st = Vt*0
#Vt[:,0] = np.random.randn(N) + Er
#st[:,0] = np.random.randn(N)

# % dynamics
for tt in range(lt-1):
    Vt[:,tt+1] = Vt[:,tt] + dt*voltage_update(Vt[:,tt], st[:,tt], volt_params, syn_params, Istim[tt])\
                 + np.random.randn(N)*np.sqrt(dt*noise)
    st[:,tt+1] = st[:,tt] + dt*synaptic_update(st[:,tt], Vt[:,tt], syn_params)

plt.figure()
plt.plot(Vt.T)

# %% Simplification (with rate model)
# %% ##########################################################################
# %% 
### time
N = 3  #three-state network
dt = 0.1  #dt
T = 1000  #10s
time = np.arange(0,T,dt)
lt = len(time)
Istim = np.zeros(lt)
Istim[4000:5000] = 1.5  #stimulus impulse

### network
Js = np.array([[10, -15, 1], [-15, 30, -5], [-50, -0, 10]])*.1  #chemical synapse
Jst = Js*1
Jg = np.zeros((3,3))*0
Jg[1,2], Jg[2,1] , Jg[0,2], Jg[2,0] = .1,.1,.1,.1  #gap junction
tau = 30
xt = np.zeros((N,lt))
rt = xt*0
xt[:,0] = np.random.randn(N)*0.1
rt[:,0] = np.tanh(xt[:,0])
noise = .05
inpt = np.array([1,0,0])
Jrt = np.zeros(lt)
Jrt[0] = Jst[0,2]
tau_s = 500   #synaptic depression rate

### dynamics
for tt in range(lt-1):
    vi,vj = np.meshgrid(xt[:,tt], xt[:,tt])
    xt[:,tt+1] = xt[:,tt] + dt/tau*(-xt[:,tt] + Jst.T @ rt[:,tt] + (Jg*(vi-vj)).sum(0) \
                 + inpt*Istim[tt]) + np.random.randn(N)*np.sqrt(dt*noise)
    Jrt[tt+1] = Jrt[tt] + dt*( -(Jrt[tt]- Js[0,2])/tau_s - max(0,.1*(1)*(rt[0,tt]+0)/tau_s) )
#    Jrt[tt+1] = max(0,Jrt[tt+1])
    rt[:,tt+1] = np.tanh(xt[:,tt+1])
    Jst[0,2] = Jrt[tt+1]
    
plt.figure()
plt.plot(rt[:,:].T)
plt.plot(Istim/max(Istim), 'k')

# %% functionize
def RFT_model():
    xt = np.zeros((N,lt))
    rt = xt*0
    xt[:,0] = np.random.randn(N)*0.1
    rt[:,0] = np.tanh(xt[:,0])
    Jrt = np.zeros(lt)
    Jrt[0] = Jst[0,2]
    for tt in range(lt-1):
        vi,vj = np.meshgrid(xt[:,tt], xt[:,tt])
        xt[:,tt+1] = xt[:,tt] + dt/tau*(-xt[:,tt] + Jst.T @ rt[:,tt] + (Jg*(vi-vj)).sum(0) \
                     + inpt*Istim[tt]) + np.random.randn(N)*np.sqrt(dt*noise)
        Jrt[tt+1] = Jrt[tt] + dt*( -(Jrt[tt]- Js[0,2])/tau_s - max(0,.1*(1)*(rt[0,tt]+0)/tau_s) )
        rt[:,tt+1] = np.tanh(xt[:,tt+1])
        Jst[0,2] = Jrt[tt+1]
    return rt

rep = 100
rec = np.zeros((2,rep))  # pre and post behavior with repeats
rec_speed = np.zeros(rep)  # for analog speed

for rr in range(rep):
    rt = RFT_model()
    rec[0,rr] = np.argmax(rt[:,4000-500:4000].sum(1))
    rec[1,rr] = np.argmax(rt[:,5000:5000+500].sum(1))
    rec_speed[rr] = rt[int(rec[0,rr]),4000] - rt[int(rec[0,rr]),4000-500]
    #rt[int(rec[0,rr]),4000-500:4000].sum()
    print(rr)

# %% analysis
pre_run = np.where(rec[0,:]==1)[0]
pre_turn = np.where(rec[0,:]==2)[0]
pr_run = len(np.where(rec[1,pre_run]==0)[0])/len(pre_run)
pr_turn = len(np.where(rec[1,pre_turn]==0)[0])/len(pre_turn)

condition=['forward','turn']
pos = np.arange(len(condition))
plt.figure()
plt.bar(pos,[pr_run, pr_turn])
plt.xticks(pos, condition)
plt.ylabel('P(reversel)',fontsize=40)

# %% forward analysis
plt.figure()
plt.plot(rec_speed[pre_run], rec[1,pre_run]==0, 'o')
plt.xlabel('effective forward velocity',fontsize=30)
plt.ylabel('reversal event (binary)',fontsize=30)
plt.xlim([-0.3,0.3])
