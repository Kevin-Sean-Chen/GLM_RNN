# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:31:47 2020

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

from pyglmnet import GLM, simulate_glm
from pyglmnet import GLMCV
from pyglmnet import GLM

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5
# %% parameters
N = 300  #number of neurons
p = 1.  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0  #learning initial constant
dt = 0.1
nsecs = 500
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
#M[sparse>p] = 0
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1

nRec2Out = N
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 0.7;
freq = 1/60;
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
#ft[ft<0] = 0
ft = ft/1.5

wo_len = np.zeros((1,simtime_len))    
zt = np.zeros((1,simtime_len))
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

# %% FORCE learning
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
        k = P @ r;
        rPr = r.T @ k
        c = 1.0/(1.0 + rPr)
        P = P - k @ (k.T * c)  #projection matrix
        
        # update the error for the linear readout
        e = z-ft[ti]
        
        # update the output weights
        dw = -e*k*c
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M = M + np.repeat(dw,N,1).T#0.0001*np.outer(wf,wo)
        #np.repeat(dw,N,1).T#.reshape(N,N).T
        #np.outer(wf,wo)
        #np.repeat(dw.T, N, 1);
        M = M*mask           

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
#x = x0
#r = np.tanh(x)
#z = z0
for t in range(len(simtime)-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    z = wo.T @ r

    zpt[0,ti] = z

zpt = np.squeeze(zpt)
plt.figure()
plt.plot(ft)
plt.plot(zpt)

# %%
###############################################################################
# %% inference
rate = rt.copy()
nneuron = 4
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
Y = np.squeeze(rate[nneuron,:])  #spike train of interest
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
stimulus = ft[:,None]  #same stimulus for all neurons
X = build_convolved_matrix(stimulus, rate.T, Ks, couple)  #design matrix with features projected onto basis functions
###pyGLMnet function with optimal parameters
distr = "binomial"
glm = GLMCV(distr=distr, tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)

# %% direct simulation
yhat = simulate_glm(distr, glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
plt.figure()
plt.plot(Y*1.,label='input')  #ground truth
plt.plot(yhat,'--',label='ouput_est')
plt.legend()

# %%
theta = glm.beta_
dc_ = theta[0]
theta_ = theta[1:]
if couple == 1:
    theta_ = theta_.reshape(nbasis,N+1)  #nbasis times (stimulus + N neurons)
    allKs = np.array([theta_[:,kk] @ Ks for kk in range(N+1)])
elif couple == 0:
    allKs = Ks.T @ theta_

plt.figure()
plt.plot(allKs.T)


# %% FORCE with GLM-net
###############################################################################
###############################################################################
# %%
def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm).T

def NL(x,spkM):
    """
    Passing x through logistic nonlinearity with spkM maximum
    """
    nl = spkM/(1+np.exp(-x))
    #nl = np.tanh(x)
    return nl

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
    #N = len(x)
    #spike = np.random.rand(N) < x*dt
    spike = np.random.poisson(x*dt)
    return spike

# %% setup
#size and length
N = 50
T = 300
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = 1.  #sparsity of connection
g = 150  # g greater than 1 leads to chaotic networks.
#Q = 10.
#E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
nbasis = 3
pad = 50
spkM = 1
tau = 1
thetas = np.random.randn(N,N,nbasis)*g*scale/1  #tensor of kernel weights
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
#Ks = Ks[0,:][None,:].T
#nbasis = 1
allK = np.zeros((N,N,pad))  #number of kernels x length of time window
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
for ii in range(N):
    for jj in range(N):
        temp = np.dot(thetas[ii,jj,:], Ks)
        temp = temp - np.mean(temp)
        allK[ii,jj,:] = temp*mask[ii,jj]

#input parameters
wo_w = np.ones((N,nbasis))
dw_w = np.zeros((N,nbasis))
wf_w = 2.0*(np.random.randn(N,nbasis)-0.5)
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 2
ft = 1*(amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    1*(amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    0*(amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft[ft<0] = 0
ft = ft*100#/1.5
 
#initial conditions
wo_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))
spks = np.zeros((N,simtime_len))

xt[:,0] = x0
rt[:,0] = NL(xt[:,0],spkM)
z = z0

rn = np.zeros((N,simtime_len,nbasis))
sn = np.zeros_like(rn)
#xn = np.zeros_like(sn)

# %% learning
P = (1.0/alpha)*np.eye(N)
P = np.repeat(P[:, :, np.newaxis], nbasis, axis=2)
for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    xt[:,tt] = ( np.einsum('ijk,jk->i',  (thetas @ Ks), spks[:,tt-pad:tt]) )
    #xt[:,tt] = (1.0-dt)*xt[:,tt-1] + dt*( np.einsum('ijk,ik->i',  (thetas @ Ks), spks[:,tt-pad-1:tt-1])/pad)
    ### spks or rt!!??
    
    spks[:,tt] = spiking( NL( (xt[:,tt]), spkM) , dt)
    rt[:,tt] = rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]/tau
    #NL( (xt[:,tt]), spkM)   ### GLM form
    #rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]  ### ODE form
    #(1.0-dt)*rt[:,tt-1]/tau + dt*spks[:,tt]   #+ np.random.randint(0,5,N)  ### weird form
    #xt[:,tt] = ( np.einsum('ijk,ik->i',  (thetas @ Ks), rt[:,tt-pad:tt]) )
    #rt[:,tt] = spiking(NL(xt[:,tt],spkM) , dt) #NL( spiking(xt[:,tt], dt), spkM) #+ np.random.randint(0,5,N)
    
    #reconstruct dynamics
    wo = wo_w @ Ks
    z = np.sum(np.einsum('ik,ik->ik', wo, rt[:,tt-pad:tt])/pad)
    
    #learning
    if np.mod(tt, learn_every) == 0:
        for nn in range(nbasis):
            tens = np.dot(thetas[:,:,nn][:,:,None],Ks[nn,:][None,:])#np.einsum('ijk,kl->ijl',thetas[:,:,nn] , Ks[nn,:])
            #xn[:,tt,nn] = (1.0-dt)*xn[:,tt-1,nn] + dt*( np.einsum('ijk,ik->i',  tens, sn[:,tt-pad-1:tt-1,nn])/pad)
            xn = np.einsum('ijk,jk->i', tens, sn[:,tt-pad-1:tt-1,nn])/pad
            #sn[:,tt,nn] = spiking( NL((xn[:,tt,nn]),spkM) , dt)
            sn[:,tt,nn] = spiking( NL((xn),spkM) , dt)
            rn[:,tt,nn] = rn[:,tt-1,nn] - dt*rn[:,tt-1,nn]/tau + dt*sn[:,tt,nn]/tau
            k = (P[:,:,nn] @ rn[:,tt,nn])[:,None];  #[:,:,nn]
            rPr = rn[:,tt,nn].T @ k
            c = 1.0/(1.0 + rPr)
            P[:,:,nn] = P[:,:,nn] - k @ (k.T * c)  #projection matrix
        
            
            #reconstruct dynamics, just for one basis projection
            wo_n = np.dot( wo_w[:,nn][:,None] , Ks[nn,:][None,:] )
            z_n = np.sum(np.einsum('ik,ik->ik', wo_n, rn[:,tt-pad-1:tt-1,nn])/pad)
            #np.einsum('ij,ij->i', wo_n, rt[:,tt-pad:tt]) #np.dot(wo_n, rt[:,tt-pad:tt])
        	
            # update the error for the linear readout
            e = z-ft[tt] ### how is error computed!
            #e = max(-10, min(10, e))
    	
        	# update the output weights
            dw = np.squeeze(-e*k*c)
            wo_w[:,nn] = wo_w[:,nn] + dw*1
            
            # update the internal weight matrix using the output's error
            thetas[:,:,nn] = thetas[:,:,nn] + np.repeat(dw[:,None],N,1).T  #.reshape(N,N).T
            #(E @ dw[None,:]).T
        
    # Store the output of the system.
    zt[tt] = np.squeeze(z)
    wo_len[tt] = np.nansum(np.sqrt(wo_w.T @ wo_w))
    
# %% plotting
plt.figure()
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.subplot(211)
plt.plot(ft)
plt.plot(zt,'--')
plt.subplot(212)
plt.plot(wo_len)

plt.figure()
plt.imshow(spks,aspect='auto')

# %% testing
zpt = np.zeros(len(simtime))
#x0 = 0.5*np.random.randn(N)
#z0 = 0.5*np.random.randn(1)
#xt = np.zeros((N,simtime_len))
#rt = np.zeros((N,simtime_len))
#spks = np.zeros((N,simtime_len))
#xt[:,0] = x0
#rt[:,0] = NL(xt[:,0],spkM)
thetas0 = np.random.randn(N,N,nbasis)*g*scale/1

for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    #xt[:,tt] = ( np.einsum('ijk,jk->i',  (thetas @ Ks), spks[:,tt-pad:tt]) )
    xt[:,tt] = (1.0-dt)*xt[:,tt-1] + dt*( np.einsum('ijk,jk->i',  (thetas @ Ks), spks[:,tt-pad-1:tt-1])/pad)
    spks[:,tt] = spiking( NL( (xt[:,tt]), spkM) , dt)
    rt[:,tt] = rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]/tau
    #rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt] 
#    xt[:,tt] = ( np.einsum('ijk,ik->i',  (thetas @ Ks), rt[:,tt-pad:tt]) )
#    rt[:,tt] = spiking(NL(xt[:,tt],spkM) , dt) #NL( spiking(xt[:,tt], dt), spkM) #+ np.random.randint(0,5,N)
    
    #reconstruct dynamics
    wo = wo_w @ Ks
    z = np.sum(np.einsum('ik,ik->ik', wo, rt[:,tt-pad:tt])/pad)
    zpt[tt] = z

plt.figure()
plt.plot(ft)
plt.plot(zpt*5)

plt.figure()
plt.imshow(spks, aspect='auto')

