# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:41:53 2021

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
N =  500  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0  #learning initial constant
dt = 0.1
nsecs = 150
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
#cutoff = 100
#uu,ss,vv = np.linalg.svd(M)
#ss[cutoff:] = 0
#M = uu @ np.diag(ss) @ vv
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
M = M*mask

nRec2Out = N
wo = np.zeros((nRec2Out,1))
pp = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 0.7;
freq = 1/60;
rescale = 5.
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
ft = ft/1.5

wo_len = np.zeros((1,simtime_len))    
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

# %% sparse, forgetful parameters
lamb = .92
### specify filter order!!! ###

eps = 1e-18
gain = np.zeros((nRec2Out,1))

def disc_func(x):
    temp = x+0
    temp[np.abs(temp)<eps] = 0
    return temp

def F_jac(x):
    temp = x+0
    temp[np.abs(temp)>=eps] = 1
    temp[np.abs(temp)<eps] = 0
    #temp = jacobian(disc_func,eps)
    return np.diag(temp.squeeze())

alpha_ = 0.0005
beta = 5
def g_beta(w,beta):
    return np.sign(w)*beta/(1+beta*np.abs(w))**2

# %% FORCE learning
zt = np.zeros((1,simtime_len))
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
for t in range(len(simtime)-1):
    ti = ti+1
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]  #xt[:,t] = x[:,0]
    z = wo.T @ r
    
    if np.mod(ti, learn_every) == 0:
#        k = P @ r
#        #P*F_jac(wo) @ r #@ F_jac(wo)
#        rPr = r.T @ k
#        #(F_jac(wo) @ r).T @ k #@ F_jac(wo)
#        c = 1.0/(1.0*1 + rPr)
#        P = 1/1*(P - k @ (k.T * c))  #projection matrix
        
        # update the error for the linear readout
        e = z-ft[ti]
        
        # update the output weights
#        dw = -e*k*c
#        wo = wo*lamb + dw
        
        ### discard function
        ### λpD,(k − 1) + F(w(k))x(k)d(k)
#        pp = lamb*pp + F_jac(wo) @ (r*ft[ti])
#        wo_ = wo*1
#        wo = P @ pp
#        dw = wo-wo_
        
        ### l1 method
#        pp = lamb*pp + ft[ti]*r
#        wo_ = wo*1
#        wo = P @ (pp - alpha_/2*g_beta(wo,beta))
#        dw = wo-wo_
        
        fs = F_jac(wo)
        ### RWLS method
        gain = 1/lamb* (P @ r) / (1+1/lamb* r.T @ P @ r)
        P = 1/lamb*(P - gain @ r.T @ P)
        dw = -e*gain
        wo = wo + dw
#        wo = disc_func(wo)
        
        # update the internal weight matrix using the output's error
        M = M + np.repeat(dw,N,1).T#0.0001*np.outer(wf,wo)
        #np.repeat(dw,N,1).T#.reshape(N,N).T
        #np.outer(wf,wo)
        #np.repeat(dw.T, N, 1);
#        M = M*mask           

    # Store the output of the system.
    zt[0,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo)	

zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft)
plt.plot(zt,'--')

plt.figure()
plt.imshow(rt,aspect='auto')
# %% testing
zpt = np.zeros((1,simtime_len))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    z = wo.T @ r

    zpt[0,ti] = z

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft,label='target')
plt.plot(zpt,label='readout')
plt.legend(fontsize=30)
print(((zpt-ft)**2).sum()/(ft**2).sum())

# %% test with off-line solution
#lamb_it = np.flip(np.array([lamb**(simtime_len-n) for n in range(simtime_len)])) #discounting
lamb_it = np.array([lamb**(simtime_len-n-1) for n in range(simtime_len)])
R = np.linalg.pinv(lamb_it[None,]*rt @ rt.T)
rr = (lamb_it*ft)[None,:] @ rt.T
w_off = R @ rr.T
test = np.linalg.pinv(rt @ rt.T) @ rt @ ft.T

plt.figure()
plt.plot(w_off, wo, 'o')
plt.xlabel('offline w',fontsize=40)
plt.ylabel('online w',fontsize=40)

# %% scanning
rep = 10
lls = np.array([0.9,0.95,0.97,0.98,0.99,1])
#np.array([0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1])
nl = len(lls)
MSE_sparse = np.zeros((nl, rep))
for rr in range(rep):
    print(rr) 
    for ll in range(nl):
        print(ll)
        lamb = lls[ll]
        
        ### intialize network
        M = np.random.randn(N,N)*g*scale
        sparse = np.random.rand(N,N)
        mask = np.random.rand(N,N)
        mask[sparse>p] = 0
        mask[sparse<=p] = 1
        M = M*mask
        wo = np.zeros((nRec2Out,1))
        
        ### training
        ti = 0
        x = x0
        r = np.tanh(x)
        z = z0
        P = (1.0/alpha)*np.eye(nRec2Out)
        for t in range(len(simtime)-1):
            ti = ti+1
            x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
            r = np.tanh(x)
            z = wo.T @ r
            
            if np.mod(ti, learn_every) == 0:                
                # update the error for the linear readout
                e = z-ft[ti]
                
                ### RWLS method
                gain = 1/lamb* (P @ r) / (1+1/lamb* r.T @ P @ r)
                P = 1/lamb*(P - gain @ r.T @ P)
                dw = -e*gain
                wo = wo + dw
        #        wo = disc_func(wo)
                
                # update the internal weight matrix using the output's error
                M = M + np.repeat(dw,N,1).T
        
        ### testing
        zpt = np.zeros((1,simtime_len))
        ti = 0
        x = x0
        r = np.tanh(x)
        z = z0
        for t in range(len(simtime)-1):
            ti = ti+1 
            
            x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
            r = np.tanh(x)
            z = wo.T @ r
        
            zpt[0,ti] = z
        
        MSE_sparse[ll, rr] = ((zpt-ft)**2).sum()/(ft**2).sum()
        
# %%
plt.figure()
plt.plot(lls, MSE_sparse,'-o',linewidth=3)
plt.xlabel(r'$\lambda$',fontsize=40)
plt.ylabel('MSE',fontsize=40)
plt.ylim([0,20])

# %%
plt.figure()
plt.plot(lls, np.mean(MSE_sparse,1),'-o',linewidth=6)
plt.errorbar(lls,np.mean(MSE_sparse,1),np.std(MSE_sparse,1),linewidth=6)
plt.plot(lls, MSE_sparse,'k*',markersize=15)
plt.xlabel(r'$\lambda$',fontsize=40)
plt.ylabel('MSE',fontsize=40)

# %% test with transition 
###############################################################################
# %%
N = 10
T = 10000
k = 2  #two states for now
xt = np.random.randn(N,T)  #network dynamics
ws = np.zeros((k,N))  #k symbolic patterns
Ss = np.sin(np.arange(0,T)/100)
Ss[Ss>=0.5] = 1
Ss[Ss<0] = 0
def ll_transition(ws, Ss, xt):
    k = ws.shape[0]
    T = len(Ss)
    exps = np.zeros((k,T))
    ll = 0
    for kk in range(k):
        exps[kk,:] = np.exp(ws[kk,:] @ xt)
    Z = exps.sum(0)
    for kk in range(k):
        pos = np.where(Ss==kk)[0]  #find states
        ll += sum(exps[kk,pos]/Z[pos])
    nll = -ll
    return nll
ll_transition(ws, Ss, xt)
    
