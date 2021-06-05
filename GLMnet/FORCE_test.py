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
N = 400  #number of neurons
p = .5  #sparsity of connection
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
# %% setup
N = 50
T = 100
dt = 0.1
time = np.arange(0,T,dt)
stim = np.random.randn(len(time))*0.1
stimulus = np.repeat(stim[:,None],3,axis=1).T #identical stimulus for all three neurons for now
nbasis = 7
pad = 100
nkernels = N**2+N  #xN coupling and N stimulus filters
thetas = np.random.randn(nkernels, nbasis)  #weights on kernels
#Ks = basis_function1(pad, nbasis)
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((nkernels,pad))  #number of kernels x length of time window
for ii in range(nkernels):
    allK[ii,:] = np.dot(thetas[ii,:], Ks)
allK = allK.reshape(N,N+1,pad)

# %% learning
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
for t in range(len(simtime)-1):
    ti = ti+1
    x = (1.0-dt)*x + M @ (r*dt)
    ## make this temporal kernel version!
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
    	M = M + np.repeat(dw,N).reshape(N,N)  #np.repeat(dw.T, N, 1);
        
    # Store the output of the system.
    zt[0,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo)	


