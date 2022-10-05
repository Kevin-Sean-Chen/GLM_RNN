# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:31:40 2022

@author: kevin
"""

import numpy as np
import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import grad, jacobian
from scipy.optimize import minimize

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

# %%
def target_dynamics(J, T, dt):
    N = J.shape[0]
    time = np.arange(0,T,dt)
    lt = len(time)
    V = np.zeros((N, lt))
    for tt in range(lt-1):
        V[:,tt+1] = V[:,tt] + dt*()
        
    return V

def BPTT(its):
    dEdx = 0
    J_inf = 0
    return J_inf

def fg_eta(eta,q,x):
    fg = eta*0 - (1-q)
    if np.isscalar(eta):
        if eta>=x:
            fg = q
    else:
        fg[eta>=x] = q
    return fg

def phi(x):
    nl = R0/(1+np.exp(-beta*(x-h)))
    return nl
# %% stochastic state network
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% parameters
N = 200
p = 2
c = 0.1
A = 3.
tau = 2
sig = 0.0
qf, qg, xf, xg = 0.65, 0.65, 1.75, 1.75
R0, beta, h = 3,1,0
zs = np.random.randn(N,p)
uu,ss,vv = np.linalg.svd(np.random.randn(N,N))
zs = (uu[:,:p]-np.mean(uu[:,:p],0))/np.std(uu[:,:p],0)
etas = phi(zs)
#etas = np.linalg.pinv(etas.T)*10
#etas = np.random.randint(0,2,size=(N,p))
temp = np.random.rand(N,N)
mask = np.zeros((N,N))
mask[temp<c] = 1
mask = 0.5*(mask+mask.T)
Js = mask*A/(N*c)* (fg_eta(etas,qf,xf) @ fg_eta(etas,qg,xg).T)
#Js = np.zeros((N,N))
#for pp in range(p):
#    for ii in range(N):
#        Js[ii,:] += mask[ii,:]*A/(N*c) * fg_eta(np.array([etas[ii,pp]]),qf,xf)*fg_eta(etas[:,pp],qf,xf)
#        for jj in range(N):
#            Js[ii,jj] += mask[ii,jj]*A/(N*c) * fg_eta(np.array([etas[ii,pp]]),qf,xf)*fg_eta(etas[jj,pp],qf,xf)
etas_ = etas[:,np.append(np.arange(1,p),0)] #permutation
Jf = 1/N* (fg_eta(etas_,qf,xf) @ fg_eta(etas,qg,xg).T)
#Jf = 1/N* (np.outer(fg_eta(etas[:,0],qf,xf),fg_eta(etas[:,1],qg,xg)) + \
#           np.outer(fg_eta(etas[:,1],qf,xf),fg_eta(etas[:,2],qg,xg)) + \
#           np.outer(fg_eta(etas[:,2],qf,xf),fg_eta(etas[:,0],qg,xg)))
#Jf = np.zeros((N,N))
#for pp in range(p):
#    for ii in range(N):
#        for jj in range(N):
#            Jf[ii,jj] += 1/(N) * fg_eta(np.array([etas_[ii,pp]]),qf,xf)*fg_eta(etas[jj,pp],qf,xf)
##        Jf[ii,:] = 1/(N*c) * fg_eta(np.array([etas_[ii,pp]]),qf,xf)*fg_eta(etas[:,pp],qf,xf)
eps_ = 0.65*1
tau_eps = 2
sig_eps = 0.65*1

T = 1000
dt = 0.1
lt = len(np.arange(0,T,dt))
ut = np.zeros((N,lt))
rt = ut*1
ept = np.zeros(lt)
ovl = np.zeros((p,lt))
patt = fg_eta(etas,qf,xf)
# %%
for tt in range(lt-1):
    ut[:,tt+1] = ut[:,tt] + dt/tau*( -ut[:,tt] + Js @ phi(ut[:,tt]) \
                  + ept[tt]*1*Jf @ phi(ut[:,tt]) + 0* etas[:,0] )\
                  + np.sqrt(2*sig**2*tau*dt)*np.random.randn(N)
    ept[tt+1] = ept[tt] + dt/tau_eps*(-ept[tt] + eps_) \
                + np.sqrt(2*sig_eps**2*tau_eps*dt)*np.random.randn()
    ovl[:,tt+1] = patt.T @ phi(ut[:,tt]) / \
                  np.sqrt(np.linalg.norm(patt,axis=0)*np.linalg.norm(phi(ut[:,tt])))
    rt[:,tt+1] = phi(ut[:,tt+1])

plt.figure()
plt.imshow(rt,aspect='auto')
plt.figure()
plt.plot(ovl.T)

# %% command small circuit
###############################################################################
# %%
states = wf_s[:N_in,:]*1
N = N_in*1
eps = 0.02
ratio = 0.5
Js = 1/N*(ratio*np.outer(states[:,0],states[:,0]) + (1-ratio)*np.outer(states[:,1],states[:,1]) \
         +.1*np.outer(states[:,0],states[:,1]))

alpha = 0.2
states = np.zeros((N,2))
temp = np.random.rand(N,2)
states[temp<alpha] = 1
Js = 1.25/((1-alpha)*alpha*N) * ((states-alpha) @ (states.T-alpha)) - 1/(alpha*N)

T = 1000
dt = 0.1
tau = 1
sig = .4
lt = len(np.arange(0,T,dt))
ut = np.zeros((N,lt))
ut[:,0] = np.random.randn(N)*1
rt = ut*1
rt[:,0] = np.random.randn(N)*.1 + states[:,0]
st_ = np.zeros((2,lt))
for tt in range(lt-1):
    ut[:,tt+1] = ut[:,tt] + dt/tau*( -ut[:,tt] + 1.0*Js @ np.tanh(ut[:,tt]) + 0*st[:,tt]@states.T) \
                 + np.sqrt(2*sig**2*tau*dt)*np.random.randn(N)
    rt[:,tt+1] = np.tanh(ut[:,tt+1])
#    Js += eps*np.outer(rt[:,tt],( ut[:,tt+1] - Js@rt[:,tt] )).T #st_[:,tt]@states.T
    temp1 = states.T @ rt[:,tt]
    temp2 = np.zeros(2)
    temp2[np.argmax(temp1)] = 1
    st_[:,tt] = temp2
    
plt.figure()
plt.imshow(rt,aspect='auto')
plt.figure()
plt.plot(st_.T)

# %%
ratios = np.arange(0,1,0.1)
es = ratios*0
for rr in range(len(es)):
    ratio = ratios[rr]
#    Js = 1/N*(ratio*np.outer(states[:,0],states[:,0]) + (1-ratio)*np.outer(states[:,1],states[:,1]) \
#         +.0*np.outer(states[:,0],states[:,1]))
    es[rr] = states[:,1].T @ Js @ states[:,1] - states[:,0].T @ Js @ states[:,0]
    
    
# %% Kinetic Ising transitions
###############################################################################
# %% function
def KIM(J,s):
    H = J @ s  #effective field
    s_ = -1*s  #a flipped spin vector
    P = np.exp(s_ @ H) / (2*np.cosh(H))  #kinetic Ising
    pr = np.random.rand(J.shape[0])
    s_[P<pr] *= -1  #flip back according to P
    return s_

def Current(h,J,s):
    theta = h + J @ s
    return theta

def Transition(si,thetai,beta):
    P = np.exp(-si*thetai*beta)/(2*np.cosh(thetai*beta))
    rand = np.random.rand()
    if P>rand:
        s_ = -si.copy()
    elif P<=rand:
        s_ = si.copy()
    return s_

def KIM2(J,s,beta):
    Theta = J@s
    s_ = s*0
    for ss in range(J.shape[0]):
        s_[ss] = Transition(s[ss], Theta[ss], beta)
    return s_

# %% parameters
N = 20
ps = np.random.randint(0,2,size=(N,2))*2-1
J = np.random.randn(N,N)*.1/N**0.5 + 1/N*(ps@ps.T)
T = 10000
freq = 0.005
simtime = np.arange(0,T)
states = np.sin(np.pi*freq*simtime)
offset = .0
states[states>offset] = 1
states[states<=offset] = -1
target = np.zeros((N,len(simtime)))
for tt in range(len(simtime)):
    if states[tt]==1:
        target[:,tt] = ps[:,0]
    elif states[tt]==-1:
        target[:,tt] = ps[:,1]
# %% learning
#eta = 0.1
#for tt in range(len(simtime)-1):
#    s_t = target[:,tt]
#    s_ = target[:,tt+1]
##    s_ = KIM(J,s_t)
#    dJ = eta*(np.outer(s_,s_t) - np.outer(np.tanh(J@s_t),s_t))
#    J = J + dJ

# MF
m = np.mean(target,1)
A = np.diag((1-m**2))
ds = target - m[:,None]
C = ds @ ds.T / ds.shape[1]
D = ds[:,1:] @ ds[:,:-1].T / ds.shape[1]
J = A @ D @ np.linalg.pinv(C)
# %% testing
beta = 1.5
s_p = target*0
s_p[:,0] = target[:,0]
for tt in range(len(simtime)-1):
    s_p[:,tt+1] = KIM2(J, s_p[:,tt], beta)#KIM(J,s_p[:,tt])
plt.figure()
plt.imshow(s_p,aspect='auto')

# %% state signal
st = np.zeros((2,len(simtime)))
for tt in range(len(simtime)):
    temp1 = ps.T @ s_p[:,tt]
    temp2 = np.zeros(2)
    temp2[np.argmax(temp1)] = 1
    st[:,tt] = temp2
    
# %% Markov testing
ps = np.random.randint(0,2,size=(N,2))*2-1
J = np.random.randn(N,N)*.1/N**0.5 + 2/N*(ps@ps.T)
# %%
beta = 1
s1 = ps[:,1]
s2 = ps[:,1]
theta = J @ s1
p_s = np.zeros(N)
for ss in range(N):
    p_s[ss] = np.exp(s2[ss]*theta[ss]*beta)/(2*np.cosh(theta[ss]*beta))
print(p_s)
print(np.prod(p_s))
