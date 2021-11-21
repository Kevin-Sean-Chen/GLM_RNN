# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 00:20:27 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.special import gammaln
import sklearn
from sklearn.metrics import r2_score
import scipy.spatial as sps

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 

# %% functions
def NL(x, lamb0, dt):
    x = lamb0*x
#    x[x<0] = 0  #ReLU
#    x = np.exp(x)  #exp
    x = 1/gain*np.log(1+np.exp(gain*x)) #soft rectification
#    x = lamb0/(1+np.exp(-x))  #sigmoid
    return x

def spiking(x, dt):
    spike = np.random.poisson(x*dt)
    return spike

def negLL(ww, S, spk, b, dt, f=np.exp, Cinv=None):
#    # if no prior given, set it to zeros
#    if Cinv is None:
#        Cinv = np.zeros([np.shape(w)[0],np.shape(w)[0]])
    N = S.shape[0]
    W = ww.reshape(N,N)
    # evaluate log likelihood and gradient
    ll = np.sum(spk * np.log(f(W @ S + b)) - f(W @ S +b)*dt) # - sp.special.gammaln(S+1) + S*np.log(dt))
    return -ll

def generative_GLM(w_map, bt, dt):
    N, lt = bt.shape[0], bt.shape[1]
    Wij = w_map.reshape(N,N)
    rt, spk, st = np.zeros((N,lt)), np.zeros((N,lt)), np.zeros((N,lt))
    for tt in range(lt-1):
        temp = NL(rt[:,tt], lamb0, dt)
        spk[:,tt] = spiking(temp, dt)
        st[:,tt+1] = st[:,tt]*(1-dt/tau) + spk[:,tt]
        rt[:,tt+1] = Wij @ st[:,tt] + bt[:,tt]
    return spk, st, rt

def infer_error(W,W_):
#    rescale = W.sum(1)/W_.sum(1)
    w_ = W_#*rescale
    delt = np.linalg.norm(W-w_)/np.linalg.norm(W)
    return delt.sum()

# %% Neural dynamics
N = 10
T = 300
dt = .1
time = np.arange(0,T,dt)
lt = len(time)
r = 1.5   #recurrent strength
tau = 1   #time scale of spike filter
lamb0 = 1  #maximum Poisson firing (corresponding to 1 spik per 1ms given dt=0.1), or gain for exp()
gain = 0.5
eps = .1  #noise strength of input
A = 2. #input strength
### tructured network
uu,ss,vv = np.linalg.svd(np.random.randn(N,N))
v1, v2, v3, v4, v5 = uu[:,0], uu[:,1], uu[:,2], uu[:,3], uu[:,4]  #orthonormal vectors
#v2 = 1*(0.5*uu[:,0] + 0.5*uu[:,1])
#v1, v2 = np.random.randn(N), np.random.randn(N)
Wij = r*(np.random.randn(N,N)/np.sqrt(N)) + 1*(np.outer(v1,v2)/N + 0*np.outer(v4,v5)/N)
#+ np.outer(v2,v2))  #low-rank
### random network
#Wij = r*np.random.randn(N,N)/np.sqrt(N)
#Wij = Wij @ Wij   #symmetry

### time-dependent input
signal = np.sin(time/20)*1.
bt_ = signal*np.ones((N,lt))*A + np.random.randn(N,lt)*eps  ### input/perturbation protocol added later

# %%
bb = 0.5*(v3)*1
bt = bt_ * bb[:,None]*1

### dynamics
rt, spk, st = np.zeros((N,lt)), np.zeros((N,lt)), np.zeros((N,lt))
for tt in range(lt-1):
    temp = NL(rt[:,tt], lamb0, dt)
    spk[:,tt] = spiking(temp, dt)   #Poisson spikes
    st[:,tt+1] = st[:,tt]*(1-dt/tau) + spk[:,tt]   #spike filtering
    rt[:,tt+1] = Wij @ st[:,tt] + bt[:,tt]   #RNN dynamics

# %%
plt.figure()
plt.imshow(st,aspect='auto')

# %% test with MLE
rr = st.copy()
Jr = 1/N*np.cov(rr)
#Cw = r/np.sqrt(N)*np.random.randn(N,N) #
Cw = Wij.T @ Wij
delta_r = 1/np.sqrt(N)*(rr*rt - rt*spk*(Wij @ rr)*dt) @ rr.T
w_ole = np.linalg.inv(Jr + np.linalg.inv(Cw)) @ delta_r

plt.figure()
plt.plot(Wij.reshape(-1), w_ole.reshape(-1),'o')

# %%
###############################################################################
# %% Inference process
# %%
dd = N*N
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, st,spk,bt,dt,np.exp),w_init,method='L-BFGS-B',tol=1e-4)
w_map = res.x
print(res.success)

# %%
plt.figure()
plt.plot(Wij.reshape(-1), w_map,'o')
plt.xlabel(r'$W_{ture}$',fontsize=45)
plt.ylabel(r'$W_{inferred}$',fontsize=45)
#plt.axis('equal')
lower = np.min(Wij)
upper = np.max(Wij)
plt.plot([lower, upper], [lower, upper], 'r--')
print('Corr:', np.corrcoef(Wij.reshape(-1),w_map)[0,1])
print('R2:', r2_score(Wij.reshape(-1), w_map))
print('Cosine:', np.sum(sklearn.metrics.pairwise.cosine_similarity(Wij, w_map.reshape(N,N))))
print('delta:',  infer_error(Wij,w_map.reshape(N,N)))

# %% evaluate Hessian
Hess = res.hess_inv.todense()
u_,s_,v_ = np.linalg.svd(Hess)
eih = s_**-1
#eih /= max(eih)
plt.figure()
plt.semilogy(eih,'-o')

# %%
###############################################################################
# %% scan over different structure strength vs. low-rank angle
reps = 10
dels = np.zeros((5,reps))  #three input vectors by repeats
cors = np.zeros((5,reps))
dd = N*N
for rr in range(reps):
    for vv in range(5):
        if vv<3:
            B = bt_*uu[:,vv][:,None]  #projection onto three basis
        elif vv==4:
            B = bt_.copy()/np.linalg.norm(np.ones(N))*A  #without projection
        elif vv==3:
            B = np.ones((N,lt))*0#np.random.randn(N,lt)*0#np.zeros_like(bt_)+1 #no input
        spk,st,_ = generative_GLM(Wij, B, dt)
        w_init = np.zeros([dd,])  #Wij.reshape(-1)#
        res = sp.optimize.minimize(lambda w: negLL(w, st,spk,B,dt,np.exp),w_init,method='L-BFGS-B',tol=1e-4)
        w_map = res.x
        
        dels[vv,rr] = infer_error(Wij, w_map.reshape(N,N))
        cors[vv,rr] = np.corrcoef(Wij. reshape(-1),w_map)[0,1]
        print(rr)

# %%
y = dels
plt.figure()
x = np.array([0,1,2,3,4])
plt.plot(x,y,'-o')
my_xticks = ['m','n','orthogonal','unit','w/o']
plt.xticks(x,my_xticks)
plt.ylabel('MSE',fontsize=50)
#plt.ylabel(r'$corr(W_{true},W_{inferr})$',fontsize=50)
plt.xlabel('input projection',fontsize=50)
plt.legend()

# %%
plt.figure()
plt.bar(x, np.mean(y,1))
cc = np.std(y,1)
plt.errorbar(x, np.mean(y,1), yerr=cc, fmt="o", color="grey", linewidth=8)
my_xticks = ['m','n','orthogonal','unit','w/o']
plt.xticks(x, my_xticks)
plt.ylabel(r'MSE($J_{inf},J_{true}$)',fontsize=60)
#plt.xlabel('input projection',fontsize=50)
plt.ylim([0.6,.8])

# %%
###############################################################################
# %%
###############################################################################
# %% debugging
#def negLL_test(ww, S):
#    N = S.shape[0]
#    W = ww.reshape(N,N)
#    # evaluate log likelihood and gradient
#    ll = np.sum(pp * (W @ S) - np.exp(W @ S))
#    return -ll
#
#ww = np.random.randn(5,5)
#ss = np.random.randn(5,100)
#gg = ww @ ss
#pp = np.random.poisson(np.exp(gg))
#res = sp.optimize.minimize(lambda w: negLL_test(w,ss),np.zeros([25,]),method='L-BFGS-B',tol=1e-8)
#w_map = res.x
#print(res.success)
#plt.figure()
#plt.plot(ww.reshape(-1), w_map,'o')
