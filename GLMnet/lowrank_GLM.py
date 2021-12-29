# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:32:52 2021

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
import timeit

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=60) 
matplotlib.rc('ytick', labelsize=60) 

# %%
def NL(x, lamb0, dt):
    x = lamb0*x
#    x[x<0] = 0  #ReLU
    x = np.exp(x)  #exp
#    x = lamb0/(1+np.exp(-x))  #sigmoid
    return x

def spiking(x, dt):
    spike = np.random.poisson(x*dt)
    return spike

def infer_error(W,W_):
#    rescale = W.sum(1)/W_.sum(1)
#    w_ = W_*rescale
    delt = np.linalg.norm(W-W_,2)/np.linalg.norm(W,2)
    return delt.sum()

def negLL(ww, S, spk, b, dt, f=np.exp, lamb=0):
#    # if no prior given, set it to zeros
#    if Cinv is None:
#        Cinv = np.zeros([np.shape(w)[0],np.shape(w)[0]])
    N = S.shape[0]
    W = ww.reshape(N,N)
    # evaluate log likelihood and gradient
    ll = np.sum(spk * np.log(f(W @ S + b)) - f(W @ S +b)*dt) - lamb*(W.T @ W).sum()
    # - sp.special.gammaln(S+1) + S*np.log(dt))
    return -ll

def generative_GLM(w_map, bt, dt):
    lamb0 = 1
    N, lt = bt.shape[0], bt.shape[1]
#    Wij = w_map.reshape(N,N)
    rt, spk, st = np.zeros((N,lt)), np.zeros((N,lt)), np.zeros((N,lt))
    for tt in range(lt-1):
        temp = NL(rt[:,tt], lamb0, dt)
        spk[:,tt] = spiking(temp, dt)
        st[:,tt+1] = st[:,tt]*(1-dt/tau) + spk[:,tt]
        rt[:,tt+1] = Wij @ st[:,tt] + bt[:,tt]
    return spk, st, rt

# %%
N = 10
T = 250
dt = .1
time = np.arange(0,T,dt)
lt = len(time)
r = .3   #recurrent strength
alpha = 1.  #structureness
tau = 1   #time scale of spike filter
lamb0 = 1  #maximum Poisson firing
### Random basis
rank = 3
randM = np.random.randn(N,N)
UU,SS,VV = np.linalg.svd(randM)
v1, v2 = UU[:,:rank], VV[:rank,:]
Wij = r* ((np.random.randn(N,N)/np.sqrt(N))*(1-alpha) + 1*(alpha)*(v1 @ v2) )
sparp = 0.
mask = np.random.rand(N,N)
Wij[mask<sparp] = 0

bb = np.random.randn(N, lt)*UU[:,5][:,None]
spk,st,_ = generative_GLM(Wij, bb, dt)

plt.figure()
plt.imshow(spk, aspect='auto')

# %% MEL
dd = N*N
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, st,spk,bb,dt,np.exp, 0.),w_init,method='L-BFGS-B',tol=1e-5)
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
print('inferr_error:', infer_error(Wij,w_map.reshape(N,N)))

# %% low-rank MAP!
def local_cov(mu,var):
    xx = np.arange(0,N)
    vv = bump(xx, mu, var)
    vv = vv/vv.sum()
    C12 = np.outer(vv,vv)
    return C12
def negll_local(kk, mu, var, phi=None):
    N = len(kk)
    nll = 0.5*N*np.log(var) + 0.5/var*((kk-mu)**2).sum()
    return nll
def negll_basis(mm, nn, Cm, Cn):
    An = nn @ np.linalg.pinv(Cn) @ nn
    An_inv = np.linalg.pinv(An)
    Sig = An_inv * Cm
    _,ls = np.linalg.slogdet(Sig)
    nll = 0.5*ls + 0.5* (mm @ np.linalg.pinv(Sig) @ mm)
    return nll

def negll_lowrank(ww, S, spk, b, dt, f=np.exp, lamb=0):
#    N = S.shape[0]
#    mm = ww[:N]
#    nn = ww[N:]
    W = UU @ np.diag(ww) @ VV #np.outer(mm,nn)
    # evaluate log likelihood and gradient
    ll = np.sum(spk * np.log(f(W @ S + b)) - f(W @ S +b)*dt) - lamb*(ww).sum()#(W.T @ W).sum()
    # - sp.special.gammaln(S+1) + S*np.log(dt))
    return -ll

# %% MEL
dd = N*1
w_init = np.zeros([dd,])  #np.concatenate((v1,v2))#Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negll_lowrank(w, st,spk,bb,dt,np.exp, 0.),w_init,method='L-BFGS-B',tol=1e-8)
w_low_map = res.x
print(res.success)

# %%
#mm_, nn_ = w_low_map[:N], w_low_map[N:]
#W_ = np.outer(mm_,nn_)
W_ = UU @ np.diag(w_low_map) @ VV
plt.figure()
plt.plot(Wij.reshape(-1), W_.reshape(-1),'o')
lower = np.min(Wij)
upper = np.max(Wij)
plt.plot([lower, upper], [lower, upper], 'r--')
print('Corr:', np.corrcoef(Wij.reshape(-1),W_.reshape(-1))[0,1])

# %% Learing basis
###############################################################################
# %%
mu1, var1, mu2, var2 = np.random.rand()*N, 2, np.random.rand()*N, 8  #two basis function with bumps
xx = np.arange(0,N)
### Bump vectors
def bump(xx, mu, var):
    temp = np.exp(-(xx-mu)**2/var/2)
    return temp/np.linalg.norm(temp)
b1, b2 = bump(xx, mu1, var1), bump(xx, mu2, var2)
def exponentiated_quadratic(xa, xb, ss, ll):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * sp.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return ss*np.exp(sq_norm/ll/2)
rankp = 3
X = np.expand_dims(np.linspace(0,N, N), 1)
#C1 = exponentiated_quadratic(X, X, .1, 3)
#C2 = exponentiated_quadratic(X, X, .1, 6)
C1 = np.eye(N) #np.outer(b1,b1) #  #np.outer(v1,v1)
C2 = np.eye(N) #np.outer(b2,b2) #  #np.outer(v2,v2)
C1 = C1/C1.sum()*.1
C2 = C2/C2.sum()*.1
#k1 = np.random.randn(rankp,N) @ np.linalg.cholesky(C1)
#k2 = np.random.randn(rankp,N) @ np.linalg.cholesky(C2)
k1, k2 = UU[:,:rankp], VV[:,:rankp]
#C1, C2 = np.cov(k1.T), np.cov(k2.T)

Wij = r* ((np.random.randn(N,N)/np.sqrt(N))*(1-alpha) + 1*(alpha)*(k1 @ k2.T) )
sparp = 0.
mask = np.random.rand(N,N)
Wij[mask<sparp] = 0

bb = np.random.randn(N, lt)*0#UU[:,5][:,None]
spk,st,_ = generative_GLM(Wij, bb, dt)

plt.figure()
plt.imshow(spk, aspect='auto')

# %%
def negll_lowrank_basis(ww, S, spk, b, dt, Cm, Cn, f=np.exp):
    N = S.shape[0]
    mm = ww[:N*rankp]
    nn = ww[N*rankp:]
    mm,nn = mm.reshape(rankp, N), nn.reshape(rankp, N)
#    sig_m = np.kron( np.linalg.pinv(nn @ np.linalg.pinv(Cn) @ nn.T), Cm)  #
#    sig_n = np.kron( np.linalg.pinv(mm @ np.linalg.pinv(Cm) @ mm.T), Cn)
    W = mm.T @ nn
    
    # evaluate log likelihood and gradient
    ll = np.sum(spk * np.log(f(W @ S + b)) - f(W @ S +b)*dt)
    
    # log prior
#    _,logdetm = np.linalg.slogdet(sig_m)
#    _,logdetn = np.linalg.slogdet(sig_n)
#    llp = -0.5*(mm.reshape(-1) @ sig_m @ mm.reshape(-1) + \
#                nn.reshape(-1) @ sig_n @ nn.reshape(-1) + logdetm + logdetn)
    llp = -0.5*np.sum(mm @ Cm @ mm.T + nn @ Cn @ nn.T)
    ll = ll + llp*1
    return -ll

# %%
# %% MEL
dd = N*2*rankp
w_init = np.random.randn(dd)  #np.concatenate((v1,v2))#Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negll_lowrank_basis(w, st,spk,bb,dt,C1,C2,np.exp),w_init,\
                           method='L-BFGS-B',tol=1e-5)
w_low_map_basis = res.x
print(res.success)

# %%
mm_, nn_ = w_low_map_basis[:N*rankp], w_low_map_basis[N*rankp:]
mm_,nn_ = mm_.reshape(rankp, N), nn_.reshape(rankp, N)
W_ = mm_.T @ nn_

plt.figure()
plt.plot(Wij.reshape(-1), W_.reshape(-1),'o')
lower = np.min(Wij)
upper = np.max(Wij)
plt.plot([lower, upper], [lower, upper], 'r--')
print('Corr:', np.corrcoef(Wij.reshape(-1),W_.reshape(-1))[0,1])
print('R2:', r2_score(Wij.reshape(-1), W_.reshape(-1)))
print('Cosine:', np.sum(sklearn.metrics.pairwise.cosine_similarity(Wij, W_)))
print('inferr_error:', infer_error(Wij, W_))

###############################################################################
# %% Comparison
###############################################################################
# %%
its = 5
lrec = np.array([250,500,1000,2000])
mls = np.zeros((len(lrec), its))
lrs = np.zeros((len(lrec), its))
mls_t = np.zeros((len(lrec), its))
lrs_t = np.zeros((len(lrec), its))
for ll in range(len(lrec)):
    bb = np.random.randn(N,lrec[ll])*0.1
    spk,st,_ = generative_GLM(Wij, bb, dt)  # take sample
    for ii in range(its):
        dd = N*N
        w_init = np.zeros([dd,])  #Wij.reshape(-1)#
        start = timeit.default_timer()
        res = sp.optimize.minimize(lambda w: negLL(w, st,spk,bb,dt,np.exp, 0.),w_init,\
                                   method='L-BFGS-B',tol=1e-5)
        stop = timeit.default_timer()
        w_map = res.x
        mls[ll,ii] = infer_error(Wij,w_map.reshape(N,N))
        mls_t[ll,ii] = stop-start
        
        dd = N*2*rankp
        w_init = np.random.randn(dd)  #np.concatenate((v1,v2))#Wij.reshape(-1)#
        start = timeit.default_timer()
        res = sp.optimize.minimize(lambda w: negll_lowrank_basis(w, st,spk,bb,dt,C1,C2,np.exp),w_init,\
                                   method='L-BFGS-B',tol=1e-5)
        stop = timeit.default_timer()
        w_low_map_basis = res.x
        mm_, nn_ = w_low_map_basis[:N*rankp], w_low_map_basis[N*rankp:]
        mm_,nn_ = mm_.reshape(rankp, N), nn_.reshape(rankp, N)
        W_ = mm_.T @ nn_
        lrs[ll,ii] = infer_error(Wij, W_)
        lrs_t[ll,ii] = stop-start
    
# %%
plt.figure()
plt.plot(lrec,mls,'k',alpha=0.5,linewidth=15)
plt.plot(lrec,lrs,'b-o',alpha=0.5,linewidth=8)
plt.xlabel('data length',fontsize=50)
plt.ylabel('MSE',fontsize=50)
plt.title('dot:low-rank ;line:MLE',fontsize=40)

# %%
plt.figure()
plt.plot(lrec,mls_t,alpha=0.5,linewidth=15)
plt.plot(lrec,lrs_t,'-o')
plt.xlabel('data length',fontsize=40)
plt.ylabel('time',fontsize=40)
plt.title('dot:low-rank ;line:MLE',fontsize=40)

###############################################################################
# %% from G-FORCE learning
###... too slow... should use torch~
###############################################################################
# %% loading rate rt and spike spks and true M_ connectivity
st = rt.copy()
bb = np.zeros_like(st)
spk = spks.copy()
# %% MLE
dd = N*N
w_init = np.zeros([dd,])  #Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negLL(w, st,spk,bb,dt,np.exp, 0.), \
                           w_init,method='L-BFGS-B',tol=1e-5,options={'maxiter':100})
M_map = res.x
print(res.success)
# %%
plt.figure()
plt.plot(M_.reshape(-1), M_map,'o')
plt.xlabel(r'$W_{ture}$',fontsize=45)
plt.ylabel(r'$W_{inferred}$',fontsize=45)

# %% low-rank
C1,C2 = np.eye(N),np.eye(N)
rankp = 2 
dd = N*2*rankp
w_init = np.random.randn(dd)  #np.concatenate((v1,v2))#Wij.reshape(-1)#
res = sp.optimize.minimize(lambda w: negll_lowrank_basis(w, st,spk,bb,dt,C1,C2,np.exp),w_init,\
                           method='L-BFGS-B',tol=1e-5,options={'maxiter':100})
M_low_map_basis = res.x
print(res.success)
# %%
mm_, nn_ = M_low_map_basis[:N*rankp], M_low_map_basis[N*rankp:]
mm_,nn_ = mm_.reshape(rankp, N), nn_.reshape(rankp, N)
M_low = mm_.T @ nn_

plt.figure()
plt.plot(M_.reshape(-1), M_low.reshape(-1),'o')
plt.xlabel(r'$W_{ture}$',fontsize=45)
plt.ylabel(r'$W_{inferred}$',fontsize=45)
