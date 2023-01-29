# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 02:05:30 2022

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:00:21 2021

@author: kevin
"""


import numpy as np
import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import grad, jacobian

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#%matplotlib qt5
# %%
def RLS(P, r):
    k = P @ r
    rPr = r.T @ k
    c = 1.0/(1. + rPr)
    P = 1/1*(P - k @ (k.T * c))  #projection matrix
    return P, k, c

def RNN(J, x, r, inpt):
    x_ = (1.0-dt)*x + J @ (r*dt) + inpt*dt
    r_ = np.tanh(x_)
    return x_, r_

# %% parameters
N = 200  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.  #learning initial constant
dt = 0.1
nsecs = 200
learn_every = 2  #effective learning rate
state_t = 10  #time steps to estimating states

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask  #sparse chaotic network

M_d = M*1  #driven network

nRec2Out = N
wo = np.zeros((nRec2Out,1))+ np.random.randn(nRec2Out,1)*0.
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)*1

wo_d = np.random.randn(nRec2Out,1)
dw_d = np.zeros((nRec2Out,1))
wf_d = wf.copy()

w_h = 2.0*(np.random.rand(N,1)-0.5)*.1

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 1
freq = 1/20;
ft_ = amp*np.sin(np.pi*freq*simtime)
high_f = amp*np.sin(np.pi*freq*simtime*4)
low_f = amp*np.sin(np.pi*freq*simtime*1)
offset = .2
pos_h = np.where(ft_>offset)[0]
pos_l = np.where(ft_<=offset)[0]
ft = np.zeros((2,simtime_len))
ft[0,pos_h] = high_f[pos_h]*1.5
ft[1,pos_l] = low_f[pos_l]*.75
#ft = ft -np.mean(ft,1)[:,None]#+ 0.
plt.figure()
plt.subplot(211)
plt.plot(ft.T)  #raw trace target  
SS = np.zeros((2,simtime_len))-0
SS[0,pos_h] = 1
SS[1,pos_l] = 1
plt.subplot(212)
plt.plot(SS.T)  #underlying states

wo_len = np.zeros((1,simtime_len))    
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))
st = np.zeros(simtime_len)

x = x0
r = np.tanh(x)
z = z0
x_d = x0
r_d = np.tanh(x_d)
z_d = 0

ft = ft.sum(0)

# % state parameters
n_state = 2
z_input = np.random.randn(n_state,N)  #state x neurons
eta_z = .1
eta_n = 1
eta_d = .01
zk = np.random.rand(n_state)#np.ones((2,1))/2
zk = zk/np.sum(zk)
nt = 0
ekt = np.zeros(n_state)
delta_k = np.zeros(n_state)
Jz = .1
Tz = .01

state_logic = 0

# %% FORCE learning
zt = np.zeros(simtime_len)
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
P_d = P*1
for t in range(len(simtime)-1):
    ti = ti+1
    ### estimation step from each state
    for ss in range(n_state):
        ### observed network
        x_, r_ = RNN(M, x, r, 0)
        z = wo.T @ r_  # linear decoding
    
#        inpt = np.array(z_input[ss,:])[:,None]  #state-input
        inpt = SS[ss,ti]  #state-input
    
        ### latent state network
        inpt_ = wf*inpt + 0*wf_d* z_d
        x_d_, r_d_ = RNN(M_d, x_d, r_d, inpt_)
        z_d = wo_d.T @ r_d_  #state-reconstruction
#        print(z_d)
        
        ekt[ss] = (z_d-ft[ti])**2  #instantaneous squared error
    
    ### error dynamics
    ekt = ekt/ekt.sum()
    delta_k = (1-eta_d*1)*delta_k + eta_d*1*ekt
    
    ### state dynamics (WTA circuit)
#    for tj in range(state_t):
#        zk = zk + dt*eta_z*(-delta_k/10**0 + Jz*zk - Tz*np.log(zk+10**-10) - nt)  #latent dynamics
#        nt = nt + -dt*eta_n*(np.sum(zk) - 1)  #norm-constraints
    
    zk = np.exp(delta_k/Tz)
    
    ### given state RNN dynamics
    Pz = zk / zk.sum()
    z_state = np.random.choice(np.arange(n_state), p=Pz) #
#    z_state = np.argmax(Pz)
#    st[ti] = zk[0]#
    st[ti] = z_state
    zkk = zk*0
    zkk[np.argmax(zk)] = 1
    zk = zkk.copy()
    
    ### observed network
    x, r = RNN(M, x, r, 0)
    rt[:,ti] = r[:,0]
    inpt = state_logic*np.array(z_input[z_state,:])[:,None]  #state-input
#    inpt = state_logic*np.array(z_input[int(SS[ss,ti]),:])[:,None]  #true states!
    z = wo.T @ r
    ### latent state network
    inpt_ = state_logic*wf*inpt + wf_d @ z#z_d
    x_d, r_d = RNN(M_d, x_d, r_d, inpt_)
    
    ### learning dynamics
    if np.mod(ti, learn_every) == 0:       
        ### RLS calculation
        P, k, c = RLS(P, r)
        P_d, k_d, c_d = RLS(P_d, r_d)
        
        ### error terms
        e = z - ft[ti]
        e_d = z_d - ft[ti]
        #e_J = M @ r - M_d @ r_d - wf*inpt #- wf_d @ stt#wo.T @ r #- z_d #*st[0,ti]#w_h*hint*0
        e_J = M @ r - M_d @ r_d - wf*ft[ti]
        
        ### update weights
        dw = -k*c*e.T
        wo = wo + dw  # target readout
        dw_d = -k_d*c_d*((1*e_d).T)
        wo_d = wo_d + dw_d
        dJ = -e_J @ (P @ r).T
        M = M + dJ    # target network

    # Store the output of the system.
    zt[ti] = np.squeeze(z)

zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft.T)
plt.plot(zt.T,'--')

plt.figure()
plt.imshow(rt,aspect='auto')

plt.figure()
plt.plot(st, '-o')

# %% testing
zpt = np.zeros((2,simtime_len*1))
Pt_inf = zpt*0  #inferred hiddne state probability
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)*1-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) + 0*wf*z_input[int(SS[0,ti]),:][:,None]
    r = np.tanh(x)
    z = wo.T @ r

    zpt[:,ti] = z.squeeze()
    e = z - ft[ti]
       

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft.T,'--',label='target')
plt.plot(zpt.T,label='readout')
#plt.legend(fontsize=30)
print(((zpt+0-ft)**2).sum()/(ft**2).sum())

#print('Corr:', np.corrcoef(zpt.sum(0), ft.sum(0))[0,1])
#plt.figure()
#hh = plt.xcorr(zpt.sum(0), ft.sum(0),maxlags=1000)
#print('lagged-corr: ', np.max(hh[1]))

# %%
plt.figure()
plt.plot(ft.sum(0),'--',label='target')
plt.plot(zpt.sum(0)+0,label='readout')
plt.legend(fontsize=20)

error_avg = sum(abs(zpt.sum(0)+0-ft.sum(0)))/simtime_len
print(['Testing MAE: ', str(error_avg.sum())]) 

plt.figure()
plt.plot(Pt_inf[:,3:].T)
plt.legend(['latent_1','latent_2'], fontsize=20)
plt.ylabel('P(state)', fontsize=40)

# %%
plt.figure()
#plt.plot(ft[0,:],'--',color='b',label='target')
#plt.plot(ft[1,:],'--',color='orange')
plt.plot(zpt[0,:],label='latent_1')#,color='blue',label='readout')
plt.plot(zpt[1,:],label='latent_2')#,color='orange')
plt.legend(fontsize=30)

# %%
f, ax = plt.subplots()
ax.plot([np.zeros(10),np.ones(10),np.ones(10)*2,np.ones(10)*3, np.ones(10)*4],\
         [force_k, gforce_k, gforce_wta_k, force_f, gforce_f],'ko',Markersize=15)
ax.plot([np.mean(force_k),np.mean(gforce_k),np.mean(gforce_wta_k),np.mean(force_f),np.mean(gforce_f)] \
         ,'o',Markersize=25,alpha=0.7)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['FORCE','FORCE-ssm','w/ WTA','FORCE w/o latent','ssm w/o latent'],)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 45)
ax.set_ylabel('R^2',fontsize=40)

# %%
f, ax = plt.subplots()
ax.plot([np.zeros(10),np.ones(10),np.ones(10)*2,np.ones(10)*3, np.ones(10)*4],\
         [force,ptforce,gforce,trforce,np.repeat(supforce,2)],'ko',Markersize=15)
ax.set_xticks([0,1,2,3,4])
ax.set_xticklabels(['FORCE','latent (w/o trans.)','w/ WTA','w/ transition','+supervised'],)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation = 45)
ax.set_ylabel('R^2',fontsize=40)
