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
# %% parameters
N = 200  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.  #learning initial constant
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
wo = np.zeros((nRec2Out,2))+ np.random.randn(nRec2Out,2)*0.
dw = np.zeros((nRec2Out,2))
#wf = 2.0*(np.random.rand(N,2)-0.5)*.1
uu,ss,vv = np.linalg.svd(np.random.randn(N,N))
#wf = np.ones((N,2))*.5
wf = uu[:,:2]*10  #10 or 0.1

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 1
freq = 1/20;
ft_ = amp*np.sin(np.pi*freq*simtime)
high_f = amp*np.sin(np.pi*freq*simtime*5)
low_f = amp*np.sin(np.pi*freq*simtime*1)
offset = .2
pos_h = np.where(ft_>offset)[0]
pos_l = np.where(ft_<=offset)[0]
ft = np.zeros((2,simtime_len))
ft[0,pos_h] = high_f[pos_h]*2
ft[1,pos_l] = -low_f[pos_l]*0.5
#ft = ft -np.mean(ft,1)[:,None]#+ 0.
plt.figure()
plt.subplot(211)
plt.plot(ft.T)  #raw trace target  
SS = np.zeros((2,simtime_len))
SS[0,pos_h] = 1
SS[1,pos_l] = 1
plt.subplot(212)
plt.plot(SS.T)  #underlying states

wo_len = np.zeros((1,simtime_len))    
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

# %%
eta_z = .01
eta_n = .1
zk = np.random.rand(2,1)#np.ones((2,1))/2
zk = zk/np.sum(zk)
nt = 0
zkk = zk*0
zkk[np.argmax(zk)] = 1
Pt = zkk*1

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
    # linear decoding
    z = (wo.T @ r)#*Pt#
#    z = (wo.T @ r)*zkk #(wo*Pt[:,0]).T @ r#
    
    if np.mod(ti, learn_every) == 0:
        k = P @ r
        rPr = r.T @ k
        c = 1.0/(1. + rPr)
        P = 1/1*(P - k @ (k.T * c))  #projection matrix
        
        # update the error for the linear readout
        e = z - ft[:,ti][:,None]#
#        e = z - ft[:,ti].sum()#
        
        # computing soft-max probability
        d_soft = (np.exp(wo[:,0] @ r)*wo[:,0] + np.exp(wo[:,1] @ r)*wo[:,1]) / \
                    sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r))
        Pt = np.array([np.exp(wo[:,0] @ r) + 0., np.exp(wo[:,1] @ r)]) #+ zk  #biased
#        Pt = np.array([np.exp(rt[:,t-1].T @ (np.outer(wf[:,0] , dw[:,0])) @ r + 0*x.T@M@r - 0*r.T@x)[0] + 0., \
#                       np.exp(rt[:,t-1].T @ (np.outer(wf[:,1] , dw[:,1])) @ r + 0*x.T@M@r - 0*r.T@x)[0]] )  # energy form!
        Pt /= sum(Pt)  #normalization
        
        ### try separation of latent z-state and regression of w here...
        # 1- add the weight or one-hot encoding of z to regression
        # 2- add dynamics of z and the WTA output layer
    
        ### bio-WTA dynamica (soft-max, normalized, persistant)
        zk = zk + -dt*eta_z*(e**2 +10*zk + nsecs*10*np.log(zk) + nt)  #latent dynamics
        nt = nt + dt*eta_n*(1 - np.sum(zk))  #norm-constraints
        
        ### gradient with states
        ### supervised
#        dw = 1*-k*c*e.T - 1*np.einsum("ijk,jk->ik", Pw, r*(SS[:,t][:,None]*(1-Pt)).T)
#        dw = 1*-k*c*e.T - 1*P @ r*(SS[:,t][:,None]*(1-Pt)*1).T  #... add reweighting here Pw:NxNx2...
        ### unsupervised
        zkk = zk*0
        zkk[np.argmax(zk)] = 1
#        dw = (1*-k*c*(1*e).T - 1*P @ r*((1-Pt)).T) #*(Pt+(Pt-Pt)).squeeze()  #gradient with latent state
#        dw = -k*c*e.T                                    # without latent
#        dw = -k*c*(Pt*e).T                               # approximate gradient weighted by probability
#        dw = -k*c*(zkk*e).T                              # approx. weighted by WTA signal
#        dw = -k*c*e.T-np.repeat(P @ r*(r-M @ r*dt),2,1)  # LDS test 
        
        ### test with iterations
#        dw = -k*c*e.T 
#        if np.mod(ti, 3) == 0:
#             dw = dw - P @ r*((1-Pt)).T
        
        ## correct transition term
        for ww in range(2):
            lamb = 1 - Pt[ww]*(1-Pt[ww])#Pt[ww]*(1 - Pt[ww])
#            fact = (r*x).T @ wf[:,ww] @ wf[:,ww][:,None].T @ (r*x)
#            fact = (wf[:,ww][:,None]*r).T @ (x @ x.T) @ (wf[:,ww][:,None]*r)
#            fact = (x.T @ wf[:,ww][:,None] @ r.T) @ (x.T @ wf[:,ww][:,None] @ r.T).T
#            fact = (rt[:,t-1].T @ wf[:,ww][:,None])**2#(x* wf[:,ww][:,None]).T @ (x* wf[:,ww][:,None])#
#            lamb = 1 - fact*Pt[ww]*(1-Pt[ww])
            k = Pw[:,:,ww] @ r
            rPr = r.T @ k
            c = 1.0/(1/lamb + 1*rPr)
            Pw[:,:,ww] = 1/1*(Pw[:,:,ww] - 1*k @ (k.T * c)) 
#            Pw[:,:,ww] = P
        dw = - np.einsum("ijk,jk->ik", Pw, r*(1*e).T + 1*r*((1-Pt)).T)
#        dw = - np.einsum("ijk,jk->ik", Pw, r*(1*e).T + (rt[:,t-1].T @ wf[:,ww][:,None] * r)*((1-Pt)).T)

        # update the internal weight matrix using the output's error
        wo = wo + dw
        M = M + wf @ dw.T
#        M = M + 0.01*r @ r.T
#        M = M + np.ones((N,2)) @ dw.T + wf @ dw12.T*1

    # Store the output of the system.
    zt[:,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo).sum()

zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(ft.T)
plt.plot(zt.T,'--')

plt.figure()
plt.imshow(rt,aspect='auto')

# %% testing
zpt = np.zeros((2,simtime_len*1))
Pt_inf = zpt*0  #inferred hiddne state probability
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)*1-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) + np.random.randn(N,1)*0.0*(wf[:,0][:,None])
    r = np.tanh(x)
    z = wo.T @ r

    zpt[:,ti] = z.squeeze()
    
#    Pt_inf[:,ti] = ( np.array([np.exp(wo[:,0] @ r), np.exp(wo[:,1] @ r)])).squeeze()#/ \
#    sum(np.exp(wo[:,0] @ r) + np.exp(wo[:,1] @ r)) ).squeeze()
    Pt_inf[:,ti] = np.array([np.exp(r.T @ (np.outer(wf[:,0] , wo[:,0])) @ r), \
          np.exp(r.T @ (np.outer(wf[:,1] , wo[:,1])) @ r)]).squeeze()
    Pt_inf[:,ti] /= sum(Pt_inf[:,ti])  #normalization
       

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft.T,'--',label='target')
plt.plot(zpt.T,label='readout')
plt.legend(fontsize=30)
print(((zpt+0-ft)**2).sum()/(ft**2).sum())

print('Corr:', np.corrcoef(zpt.sum(0), ft.sum(0))[0,1])
plt.figure()
hh = plt.xcorr(zpt.sum(0), ft.sum(0),maxlags=1000)
print('lagged-corr: ', np.max(hh[1]))

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
