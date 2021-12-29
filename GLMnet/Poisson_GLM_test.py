# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 12:16:48 2021

@author: kevin
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp  #.optimize.minimize, .special.gammaln, and .linalg.hankel
from scipy.linalg import hankel
from scipy.optimize import minimize
from scipy.special import gammaln

# %% load parameters
# simulate Poisson GLM with autoregressive stimulus filter
w = np.array([ 2.5, 0.0, 0.0, 0.0, -0.01368046, -0.01986828,
 -0.03867417, -0.05218188, -0.10044614, -0.12434759, -0.19540891, -0.23327453,
 -0.32255702, -0.40001292, -0.46124429, -0.46235415, -0.43928836, -0.52066692,
 -0.58597496, -0.15804368,  1.2849799,   1.91338741,  1.10402054,  0.23188751,
  0.00331092, -0.0111924, ])

D = len(w)  #kernel length
T = 10000  #time bins
xx = np.random.randint(0,2,[T,])-0.5  #binary stimuli
xx = 0.3*np.random.randn(T)  #Gaussian stimuli
dtsp = 0.01  #spiking bin size

# %% Generate spikes

def PGLM_spk(xx,w,dtsp):
    D = len(w)
    T = len(xx)
    # generate design matrix 
    X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
    X = np.concatenate((np.ones([T,1]),X),axis=1)
    # generate spikes 
    y = np.random.poisson(np.exp(X @ w)*dtsp)
    return y, X

y, X = PGLM_spk(xx,w,dtsp)
# look at data
plt.figure()
plt.subplot(2,1,1)
plt.plot(xx[:100])
plt.xlabel("time bin")
plt.ylabel("stimulus value")
plt.subplot(2,1,2)
plt.plot(y[:100])
plt.xlabel("time bin")
plt.ylabel("spike count")

# %% log ll function!
def poisson_log_like(w,Y,X,dt,f=np.exp,Cinv=None):
    """
    Poisson GLM log likelihood.
    f is exponential by default.
    """
    # if no prior given, set it to zeros
    if Cinv is None:
        Cinv = np.zeros([np.shape(w)[0],np.shape(w)[0]])

    # evaluate log likelihood and gradient
    ll = np.sum(Y * np.log(f(X@w)) - f(X@w)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt)) + 0.5*w.T@Cinv@w

    # return ll
    return ll

# %% BASIS function here!!!
    
###############################################################################
###############################################################################
    
B = 8
D = len(w)
basis_set = (basis_function1(D,B).T) #flipkernel
Y, X = design_matrix(0, xx[:,None], y[:,None], D, 0, basis_set)
dd = X.shape[1]
lambda_ridge = np.power(2.0,4)
lambda_ridge = 0.0
Cinv = lambda_ridge*np.eye(dd)
Cinv[0,0] = 0.0 # no prior on bias
# fit with MAP
res = sp.optimize.minimize(lambda w: -poisson_log_like(w,Y,X,dtsp,np.exp,Cinv), np.zeros([dd,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
w_map = res.x

plt.figure()
plt.plot(w)
plt.plot(flipkernel(w_map[1:] @ basis_set))

# %%

###############################################################################
###############################################################################

# %% inference
# prior
lambda_ridge = np.power(2.0,4)
lambda_ridge = 0.0
Cinv = lambda_ridge*np.eye(D)
Cinv[0,0] = 0.0 # no prior on bias
# fit with MAP
res = sp.optimize.minimize(lambda w: -poisson_log_like(w,y,X,dtsp,np.exp,Cinv), np.zeros([D,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
w_map = res.x

# %%
plt.figure()
plt.plot(w)
plt.plot(w_map,'--')

# %% MAP estimate for decoding
def make_designX(xx,D):
    """
    Return design matrix TxD, with time T and window D, given time series xx
    """
    T = len(xx)
    X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
    X = np.concatenate((np.ones([T,1]),X),axis=1)
    return X

def MAP_decoding(xx,w_map,Y,mu_x,sig_x,dt,f=np.exp,Cinv=None):
    """
    Poisson GLM log likelihood.
    f is exponential by default.
    """
    # if no prior given, set it to zeros
    if Cinv is None:
        Cinv = np.zeros([np.shape(w_map)[0],np.shape(w_map)[0]])
    
    # make design matrix X
    D = len(w_map)
    X = make_designX(xx,D)
    # evaluate log likelihood and gradient
    ll = np.sum(Y * np.log(f(X @ w_map)) - f(X @ w_map)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt)) \
    + 0.5*w_map.T @ Cinv @ w_map + 0.5*(xx-mu_x)[:,None].T @ (xx-mu_x)[:,None]*(sig_x)**-1

    # return ll
    return ll

# prior, emperical
mu_x = np.mean(xx)
sig_x = np.cov(xx)
# fit with MAP
res = sp.optimize.minimize(lambda x: -MAP_decoding(x,w_map,y,mu_x,sig_x,dtsp,np.exp,Cinv), \
                           np.zeros([T,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
xx_map = res.x

# %% 
plt.figure()
plt.plot(xx_map)
plt.plot(xx*0.5,'--')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% GLM-network
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#give input and spike train
y = spk.T  #NxT
xx = np.repeat(stim[None,:],3,axis=0).T  #Iapp.T

# %%
def design_matrix(idd, stim, spk, pad, cp_mode, basis_set=None):
    """
    idd:      int for neuron id
    stim:     TxN stimuli
    spk:      TxN spiking pattern
    pad:      int for kernel width, or number of weight parameters
    cp_mode:  binary indication for coupling or independent model
    basis_set:DxB, where D is the kernel window and B is the number of basis used
    """
    
    T, N = spk.shape  #time and number of neurons
    xx = stim[:,idd] #for input stimuli
    D = pad  #pad for D length of kernel
    y = spk  #spiking patterns
    #y = spk[:,idd] #for self spike
    #other_id = np.where(np.arange(N)!=idd)
    #couple = spk[:,other_id]
    
    if basis_set is None:
        if cp_mode==0:
            X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:]) #make design matrix
            X = np.concatenate((np.ones([T,1]),X),axis=1)  #concatenate with constant offset
        elif cp_mode==1:
            X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
            for nn in range(N):
                yi = y[:,nn]
                Xi = sp.linalg.hankel(np.append(np.zeros(D-2),yi[:T-D+2]),yi[T-D+1:])  #add spiking history
                X = np.concatenate((X,Xi),axis=1)
            X = np.concatenate((np.ones([T,1]),X),axis=1)
    else:
        basis = flipkernel(basis_set[:,1:].T)  #the right temporal order here!
        if cp_mode==0:
            X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
            X = X @ basis  #project to basis set
            X = np.concatenate((np.ones([T,1]),X),axis=1)
        elif cp_mode==1:
            X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
            X = X @ basis
            for nn in range(N):
                yi = y[:,nn]
                Xi = sp.linalg.hankel(np.append(np.zeros(D-2),yi[:T-D+2]),yi[T-D+1:])
                Xi = Xi @ basis
                X = np.concatenate((X,Xi),axis=1)
            X = np.concatenate((np.ones([T,1]),X),axis=1)      
        
    y = spk[:,idd]
    return y, X

idd = 1
ww = 100  #pad window size
Y, X = design_matrix(idd, xx, y, ww, 1)
dd = X.shape[1]
# %% inference test
lambda_ridge = np.power(2.0,4)
lambda_ridge = 0.0
Cinv = lambda_ridge*np.eye(dd)
Cinv[0,0] = 0.0 # no prior on bias
# fit with MAP
res = sp.optimize.minimize(lambda w: -poisson_log_like(w,Y,X,dtsp,np.exp,Cinv), np.zeros([dd,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
w_map = res.x

# %% kernels
base = w_map[0]
Ks = np.reshape(w_map[1:],[int(len(w_map[1:])/(ww-1)),ww-1])
plt.figure()
plt.plot(Ks.T)

# %% with smooth basis functions
B = 8
D = ww
basis_set = (basis_function1(D,B).T) #flipkernel
#ww = D
Y, Xb = design_matrix(idd, xx, y, ww, 1, basis_set)
dd = Xb.shape[1]  # should eqal to 1+B*(N+1)
# %%
lambda_ridge = np.power(2.0,4)
lambda_ridge = 0.0
Cinv = lambda_ridge*np.eye(dd)
Cinv[0,0] = 0.0 # no prior on bias
# fit with MAP
res = sp.optimize.minimize(lambda w: -poisson_log_like(w,Y,Xb,dtsp,np.exp,Cinv), np.zeros([dd,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
w_map = res.x
print(w_map)
# %%
base = w_map[0]
Ks = np.reshape(w_map[1:],[int(len(w_map[1:])/B),B])
Ks = Ks @ basis_set
plt.figure()
plt.plot(Ks.T)

# %%
def Poisson_GLM(w,X,dt):
    """
    w:  D length weight vector 
    X:  TxD design matrix
    dt: time steps for a bin
    """
    fx = np.exp(X @ w)
    spk = np.random.poisson(fx*dt)
    return spk

y_rec = Poisson_GLM(w_map, Xb, dtsp)

plt.figure()
plt.plot(Y)
plt.plot(y_rec+2,'--')

# %% reconstruct with kernels
#Ks = flipkernel(Ks)
w_rec = Ks[:,:-1].reshape(-1)
w_rec = np.append(w_rec, base)

y_rec = Poisson_GLM(w_rec, X, dtsp)

plt.figure()
plt.plot(Y)
plt.plot(y_rec+2,'--')

# %% Hessian analysis
hess = res.hess_inv.matmat(np.eye(dd))
#hess = np.linalg.pinv(hessi)
uu,vv = np.linalg.eig(hess)
plt.figure()
plt.plot(uu,'-o')

# %%
eig = 1
w_test = np.real(vv[:,eig])
y_rec = Poisson_GLM(w_test, X, dtsp)
plt.figure()
plt.plot(Y)
plt.plot(y_rec+2,'--')
