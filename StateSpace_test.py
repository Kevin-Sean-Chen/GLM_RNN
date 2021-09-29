# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 07:35:48 2021

@author: kevin
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy as sp
from scipy.optimize import minimize

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% Model: Poisson SSM
N = 3
vv = np.random.randn(N)
A = np.outer(vv,vv)#np.random.randn(N,N)
A = A/np.sum(A,axis=1)
Cq, C1 , Cy= np.zeros((N,N)), np.zeros((N,N)), np.zeros((N,N))
np.fill_diagonal(Cq,np.random.randn(N))
mu1 = np.random.randn(N)
np.fill_diagonal(C1,np.random.randn(N))
np.fill_diagonal(Cy,np.random.randn(N))
B = np.random.randn(N)
def poissonNL(x):
    return np.random.poisson(np.exp(x))
### generative
T = 1000
Qs = np.zeros((N,T))
Ys = np.zeros((N,T))
Qs[:,0]
for tt in range(1,T):
    Qs[:,tt] = A @ Qs[:,tt-1] + Cq @ np.random.randn(N)
    Ys[:,tt] = B*Qs[:,tt] + Cy @ np.random.randn(N)
    #poissonNL(B * Qs[:,tt])
    
# %% Inference:
vv = np.zeros(T*N)
Cq_, C1_ , Cy_ = np.linalg.pinv(Cq), np.linalg.pinv(C1), np.linalg.pinv(Cy)
vv[0], vv[-1] = (C1_ + A.T @ Cq_ @ A - B.T @ Cy_ @ B),  (A.T @ Cq_ @ A - B.T @ Cy_ @ B)
#vv[1:-1] = np.ones(T-2)*(A.T @ Cq_ @ A - B.T @ Cy_ @ B)
H = np.diag(vv)  #block diagonal for N-dim
d = B.T @ Cy_ @ Ys
d[0] = d[0] + C1_ @ mu1

# %% test
Q_opt = -np.linalg.pinv(H) @ d #direct optimization

###
# hight dim
# non-Gaussian
# optimization for Q vs. theta

###############################################################################
# %% continuous latent test  (Logistic Gaussian Process)
###############################################################################
# %% generative model
dt, T = 0.1, 50
time = np.arange(0,T,dt)
lt = len(time)
eta_z, tau_z = .1, 10
K = 0.01
def sigmoid(x):
    return 1/(1+np.exp(-x))
def observe_y(x,p):
    if p > np.random.rand():
        beta = 1
    else:
        beta = 0
    #wv = x + K*np.random.randn()
    #rt = beta*np.random.randn()*10
    out = beta  #wv+rt
    return out
x = np.sin(time/10)*0
z, y = np.zeros(lt), np.zeros(lt)
for tt in range(lt-1):
    z[tt+1] = z[tt] + dt*(-z[tt]/tau_z) + eta_z*np.random.randn()
    y[tt] = observe_y(x[tt],sigmoid(z[tt]))

# %% GP
#def Kernel(x,xx):
#    return ss*np.exp(-(x-xx)**2/ll)
#ss = 1
#ll = 1000
#K_ = np.zeros((lt,lt))
#for ii in range(lt):
#    for jj in range(lt):
#        K_[ii,jj] = Kernel(ii,jj)
        
# %% faster way of generative model
#kSE = @(r,l,x)(r*exp(-(bsxfun(@plus,x(:).^2,x(:).^2')-2*x(:)*x(:)')/(2*l.^2)));
# Define the exponentiated quadratic 
def GP_exp(xa, xb, ss,ll):
    """Exponentiated quadratic  with σ=1"""
#    sq_norm = -ss * sp.spatial.distance.cdist(xa, xb, 'sqeuclidean')
#    return np.exp(sq_norm/ll)
    sqdist = np.sum(xa**2, 1).reshape(-1, 1) + np.sum(xb**2, 1) - 2 * np.dot(xa, xb.T)
    return ss**2 * np.exp(-0.5 / ll**2 * sqdist)
ss = 1
ll = 10
t_ = np.expand_dims(np.arange(0,lt),1)
K_ = GP_exp(t_,t_,ss,ll)

z = K_ @ np.random.randn(lt)
y = np.zeros(lt)
y[sigmoid(z)>np.random.rand(lt)] = 1

# %% inference
# %% MAP
niter = 50
f = K_ @ np.random.randn(lt)
#ss,ll = k_map
#K_ = GP_exp(t_,t_,ss,ll)
Im = np.diag(np.ones(lt))
for nn in range(niter):
    u = sigmoid(f)
    #np.exp(f)/np.sum(np.exp(f))
    WW = 1*(np.diag(u)  np.outer(u,u))
    u_,s_,v_ = np.linalg.svd(WW)
    RR = u_ @ np.diag(s_**0.5)  #Cholesky decomposition via svd
#    RR = (lt**0.5)*(np.diag(u**0.5) - np.outer(u,u)@np.diag(u)**(-0.5))
#    WW = RR @ RR.T
    dlogP = y-sigmoid(f) #1/(np.exp(f)+1)   #derivitive
    v = WW @ f + dlogP
    IRKR = Im + RR.T@K_@RR
    uu,ss,vv = np.linalg.svd(IRKR)
    inv_TRKR = vv.T @ np.diag(ss**-1) @ uu.T  #inverse with svd
    f = K_ @ (Im - RR @ inv_TRKR @RR.T@K_) @ v  #Newton method
    print(nn)
    
# %% Algorithm3
niter = 100
f = K_ @ np.random.randn(lt)
#ss_marg,ll_marg = k_map
K_ = GP_exp(t_,t_,ss_marg,ll_marg)
Im = np.diag(np.ones(lt))
for nn in range(niter):
    u = sigmoid(f)
    WW = 1*(np.diag(u) - np.outer(u,u))
    u_,s_,v_ = np.linalg.svd(WW)
    RR = u_ @ np.diag(s_**0.5)  #Cholesky decomposition via svd
    IWKW = Im + RR.T @ K_ @ RR
    LL = np.linalg.cholesky(IWKW)
    dlogP = y-sigmoid(f) #1/(np.exp(f)+1)   #derivitive
    v = WW @ f + dlogP
    WKb = RR.T @ K_ @ v
    a = v - RR.T @ np.linalg.solve(LL.T, np.linalg.solve(LL, WKb))  #matrix inversion method
    f = K_ @ a
    print(nn)

# %%
plt.figure()
plt.plot(z,label='true latent')
plt.plot(f-np.mean(f)*0,'--',label='inferred GP')
plt.legend()

# %% Laplace
ct_ = 0.0001
def approx(pars,f):
    ss,ll = pars
    K_ = GP_exp(t_,t_,ss,ll)
    uu,s_,vv = np.linalg.svd(K_)   #make low-rank approx here~~~
    cut = np.sum(s_>ct_)
    Kinv = vv[:,:cut] @ np.diag(s_[:cut]**-1) @ uu[:,:cut].T  #low-rank with SVD inversion
    ### Kinv = vv.T @ np.diag(s_**-1) @ uu.T
#    K_ = uu[:,:cut] @ np.diag(s_[:cut]) @ uu[:,:cut].T  #low-rank approximation
#    LL = np.linalg.cholesky(K_)  #via Cholesky method
#    alpha = np.linalg.solve(LL.T, np.linalg.solve(LL,f))
#    ql = -0.5*np.dot(alpha,f) -np.sum(np.diag(LL)) + (np.dot(y,f)+lt*np.log(np.sum(np.exp(f)))) - logdet
    (sign, logdet) = np.linalg.slogdet(Im + RR.T@K_@RR)
    ql = -0.5*f.T @ Kinv @ f + np.sum(-np.log(1+np.exp(-np.dot(y,f)))) - 0.5*logdet
    ### (np.dot(y,f)+lt*np.log(np.sum(np.exp(f))))
    return -ql

res = sp.optimize.minimize(lambda sl: approx(sl,f), np.array([2,9]), bounds = ((1e-3,None),(1e-3,None)),\
                           method='L-BFGS-B', options={'disp': True})
k_map = res.x
