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
# %% parameters
N =  300  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0  #learning initial constant
dt = 0.1
nsecs = 200
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask

nRec2Out = N
wo = np.zeros((nRec2Out,2))
dw = np.zeros((nRec2Out,2))
wf = 2.0*(np.random.rand(N,2)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 1
freq = 1/30;
ft_ = amp*np.sin(np.pi*freq*simtime)
high_f = amp*np.sin(np.pi*freq*simtime*4)
low_f = amp*np.sin(np.pi*freq*simtime*1)
offset = 0.3
pos_h = np.where(ft_>offset)[0]
pos_l = np.where(ft_<=offset)[0]
ft = np.zeros((2,simtime_len))
ft[0,pos_h] = high_f[pos_h]
ft[1,pos_l] = low_f[pos_l]
plt.figure()
plt.plot(ft.T)  #raw trace target  
SS = np.zeros((2,simtime_len))
SS[0,pos_h] = 1
SS[1,pos_l] = 1
plt.plot(SS.T)  #underlying states

wo_len = np.zeros((1,simtime_len))    
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0


# %% FORCE learning
zt = np.zeros((2,simtime_len))
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
Pw = np.concatenate((P[:,:,None],P[:,:,None]),2)
for t in range(len(simtime)-1):
    ti = ti+1
    x = (1.0-dt)*x + M @ (r*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]
    z = wo.T @ r
    
    if np.mod(ti, learn_every) == 0:
        k = P @ r
        rPr = r.T @ k
        c = 1.0/(1. + rPr)
        P = 1/1*(P - k @ (k.T * c))  #projection matrix
        
        # update the error for the linear readout
        e = z-ft[:,ti][:,None]
        
        # computing soft-max probability
        d_soft = (np.exp(wo[:,0] @ r)*wo[:,0] + np.exp(wo[:,1] @ r)*wo[:,1]) / \
                    sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))
        Pt = np.array([np.exp(wo[:,0] @ r), np.exp(wo[:,1] @ r)])/sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))
        
        # with IRLS for weightings
        for ww in range(0,2):
            kw = Pw[:,:,ww] @ r
            rPrw = r.T @ kw
            lamb_w = np.exp(wo[:,ww] @ r) / sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))
            cw = 1.0/( lamb_w + rPrw)
            Pw[:,:,ww] = Pw[:,:,ww] - kw @ (kw.T * cw)  #projection matrix
#            rw = np.exp(wo[:,ww] @ r) / sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))
#            Pw[:,:,ww] = Pw[:,:,ww] - Pw[:,:,ww] @ r*1 @ r.T @ Pw[:,:,ww] / (1+ r.T @ Pw[:,:,ww] @ r*1)
        
        ### gradient with states
#        dw = 1*-k*c*e.T - 1*np.einsum("ijk,jk->ik", Pw, r*(SS[:,t][:,None]*(1-Pt)).T)
        dw = 1*-k*c*e.T - 1*P @ r*(SS[:,t][:,None]*(1-Pt)*Pt).T  #... add reweighting here Pw:NxNx2...
        
        ### gradient
#        dw = -k*c*e.T - 0.000*r*(SS[:,t][:,None] - Pt).T
        ### diference
#        dw = -k*c*e.T - 0.001*r @ (SS[:,t][:,None]*Pt).T #+ P @ (wo - d_soft[:,None])
        
        # test with Hesisan calculation for soft-max
#        H_diag = r*(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))*(np.exp(wo[:,0] @ r)) - np.exp(wo[:,1] @ r)**2
#        H_norm = r / (np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))**2
#        H_off = np.exp(wo[:,0] @ r)*np.exp(wo[:,1] @ r)
#        H_off = H_off*np.ones((N,N))
#        H_off = H_off - H_off.diagonal()
#        Hess_soft = (np.diag(H_diag[:,0]) + H_off)*H_norm
#        
#        dw = -((P + Hess_soft) @ r)*e.T #.... wrong! the whole inverse matrix should change...
        
#        P = 1/1*(P - k @ (k.T * c))  #later update inverse matrix
        
        # update the internal weight matrix using the output's error
        wo = wo + dw
        M = M + wf @ dw.T          

    # Store the output of the system.
    zt[:,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo).sum()

zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft)
plt.plot(zt.T,'--')

plt.figure()
plt.imshow(rt,aspect='auto')

# %% testing
zpt = np.zeros((2,simtime_len))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt)
    r = np.tanh(x)
    z = wo.T @ r

    zpt[:,ti] = z.squeeze()

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft.T,'--',label='target')
plt.plot(zpt.T,label='readout')
plt.legend(fontsize=30)
print(((zpt-ft)**2).sum()/(ft**2).sum())

plt.figure()
plt.plot(ft.sum(0),'--',label='target')
plt.plot(zpt.sum(0),label='readout')
plt.legend(fontsize=30)
