# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:41:25 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import itertools
from scipy.signal import correlate

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

# %% Implement simple EP
###############################################################################
# %% functions
def energy(x,W,b):
    s = NL(x)
    E = -0.5*s.T @ W @ s + b.T @ s
    return E

def cost(x,y,beta):
    s = NL(x)
    C = beta*(s-y)**2
    return C

def NL(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x#np.tanh(x)

# %%
N = 50
lt = 1000
W = np.random.randn(N,N)
y = np.random.randint(0,2,N)
s = np.random.randint(0,2,N)
x = np.random.randn(N)

equ_step = 10
clamp_step = 10
for tt in range(lt):
    for te in range(equ_step):
        x = W @ s
        s = NL(x)
    for tc in range(clamp_step):
        break ###
        
# %% Implement FDT in Hopfield network
###############################################################################
# %% functions
def sigma(h,T):
    """
    Stochastic nonlinearity following Boltzmann machine
    """
    P = 1/(1+np.exp(-2*h/T))  #from the mean field results
    ll = len(h)
    rand = np.random.rand(ll)
    pos = np.where(P-rand>0)[0]
    r = -np.ones(ll)
    r[pos] = +1
    return r

def spin_corr(St,win,mode=None):
    """
    Measure spin correlation across time, with NxT data St and window size win,
    output is <C(t,t+win)>_T
    """
    N,T = St.shape
    Ct = np.zeros((N,N,win)) ##
    mu = np.mean(St,axis=1)
    if mode is None:
        for ww in range(win):
            ct = 0
            for tt in range(T-win):
                ct += np.outer(St[:,tt]-mu, St[:,tt+ww]-mu)  #cross-corr
                #(St[:,tt]-mu)*(St[:,tt+ww]-mu)  #auto-corr
                #np.dot(St[:,tt],St[:,tt+ww])/N
            Ct[:,:,ww] = ct/T ##
    else:
        sample = mode
        ti = np.random.randint(0,T-win,sample)
        for ww in range(win):
            ct = 0
            for tt in ti:
                ct += np.outer(St[:,tt]-mu, St[:,tt+ww]-mu)  #cross-corr
            Ct[:,:,ww] = ct/sample ##
        
    return Ct

def HopfieldNet(N,P,M=None):
    """
    Generate Hopfield network connectivity give N neurons and P patterns
    with mode==None the patterns are random. Otherwise feed in patterns constructed in a list
    """
    J = np.zeros((N,N))
    if M==None:
        for pp in range(P):
            eps = np.random.randint(0,2,N)
            eps[eps==0] = -1
            J += np.outer(eps,eps)
    else:# M = patterns(N,P,'random') out side of the function
        for pp in range(P):
            eps = M[pp]
            J += np.outer(eps,eps)
    return J

def patterns(N,P,mode):
    """
    With N spins, return P (P<N^2) unique spins configurations
    )
    """
    if mode=='iterative':  #only works when N is small enough (N<20) for enumeration
        spins = list(itertools.product([-1, 1], repeat=N))    
        pp = np.random.choice(2**N,P,replace=False)
        eps = list([spins[p] for p in pp])
    if mode=='random':
        eps = [2*(np.random.randint(0,2,N)-0.5) for p in range(P)]
    return eps

def NeuralDynamics(N,T,pars, connect=None):
    """
    Give N neurons T time and pars: w_rand weight for random and w_strc for structure, P patterns with mode
    return the neural activity generated from Hopfield framwork
    """
    kbT, w_rand, w_strc, P, mode = pars[0], pars[1], pars[2], pars[3], pars[4], 
    
    St = np.zeros((N,T))
    S0 = np.random.randint(0,2,N)
    S0[S0==0] = -1
    St[:,0] = S0
    Jn = np.random.randn(N,N)
    M = patterns(N,P,mode)
    
    if connect is None:
        J = w_strc*HopfieldNet(N,P,M) + w_rand*Jn
    else:
        J = connect
        
    for tt in range(1,T):
        ht = J @ St[:,tt-1]
        St[:,tt] = sigma(ht,kbT)
    return St, M, J

# %% dynamics ---- correlation
N = 60
T = 10000

kbT, w_rand, w_strc, P, mode = 1., 1., .1, 3, 'random'
pars = kbT, w_rand, w_strc, P, mode
St, M, J = NeuralDynamics(N,T,pars)

plt.figure()
plt.imshow(St,aspect='auto')
plt.xlabel('time steps',fontsize=45)
plt.ylabel('cells',fontsize=45)

# %%
eps = patterns(N,2,'random')
J = np.outer(eps[0],eps[0]) + np.outer(eps[0],eps[1]) + np.outer(eps[1],eps[1]) + np.random.randn(N,N)*10
St, M, J = NeuralDynamics(N,T,pars, connect=J)

plt.figure()
plt.imshow(St,aspect='auto')
plt.xlabel('time steps',fontsize=45)
plt.ylabel('cells',fontsize=45)

# %%
win = 50
Ct_s = spin_corr(St,win,1000)
plt.figure()
plt.plot(Ct_s.reshape(N**2,win).T,'-o')
plt.xlabel(r'$\delta t$',fontsize=45)
plt.ylabel(r'$C_t$',fontsize=45)

# %% dynamics ---- perturbation
delta = 0.1
rep = 1000
pxs = np.zeros((N,win))
dxs = np.zeros((N,N,win)) ##
du = 2*(np.random.randint(0,2,N)-0.5)
mu = np.mean(St,axis=1)
#M = patterns(N,P,'random')

for rr in range(rep):
#    ti = np.random.randint(T-win)
#    Si = St[:,ti]
    pi = np.mod(rr,P) #iteratively select the pattern of interest for perturbation
    du = M[pi]
    #du = 2*(np.random.randint(0,2,N)-0.5)
    purb = mu + du*delta
    ht = J @ mu +  du*delta  #J @ purb
    for ww in range(win):
        pxs[:,ww] = sigma(ht,kbT)
        ht = J @ pxs[:,ww] #+ du*delta
        dxs[:,:,ww] += np.outer( (pxs[:,0]-mu) , (pxs[:,ww]-mu) ) / rep
    #J += np.outer(pxs[:,ww],pxs[:,ww])/rep  #learning test!
    #dxs[:,:,rr] -= np.repeat(mu[:,None],win,1)

# %%
Cov = np.cov(St)
#bdxd = np.mean(dxs,axis=2)
dec_dx = np.zeros((N,N,win))
for ww in range(win):
    dec_dx[:,:,ww] = Cov @ dxs[:,:,ww]
#test = np.einsum('ij,jik->ijk',Cov, dxs)
plt.figure()
plt.plot(dec_dx.reshape(N**2,win).T,'-o')
plt.xlabel(r'$\delta t$',fontsize=45)
plt.ylabel(r'$\chi_t$',fontsize=45)

# %%
plt.figure()
plt.plot(Ct_s.reshape(-1), dec_dx.reshape(-1),'o')
plt.xlabel(r'$C_t$',fontsize=45)
plt.ylabel(r'$\chi_t$',fontsize=45)

# %%
# HYPOTHESIS: regression for better W learning??
ddt = 10
dw = (Ct_s[:,:,:ddt] - dec_dx[:,:,:ddt])
J_ = np.mean(dw,axis=2)/delta

# %% training
ddt = 10
epoch = 5
delta = 1
J_ = J.copy()
for ee in range(epoch):
    ### neural dynamics
    #kbT, w_rand, w_strc, P, mode = .5, 1, 0., 5, 'random'
    #pars = kbT, w_rand, w_strc, P, mode
    St, _, _ = NeuralDynamics(N,T,pars,J_)
    ### correlation
    Ct_s = spin_corr(St,win,1000)
    mu = np.mean(St,axis=1)
    dxs = np.zeros((N,N,win))
    ### perturbation
    for rr in range(rep):
        pi = np.mod(rr,P)  #iteratively select from the set of patterns
        pi = np.random.choice(P,1)[0]
        du = M[pi]
        purb = mu + du*delta
        ht = J_ @ mu +  (du)*delta  #J @ purb
        for ww in range(win):
            pxs[:,ww] = sigma(ht,kbT)
            ht = J_ @ pxs[:,ww] #+ du*delta
            dxs[:,:,ww] += np.outer( (pxs[:,0]-mu) , (pxs[:,ww]-mu) ) / rep
    Cov = np.cov(St)
    dec_dx = np.zeros((N,N,win))
    for ww in range(win):
        dec_dx[:,:,ww] = Cov @ dxs[:,:,ww]
        
    ### learning/matching/EP
    #dw = (dec_dx[:,:,:ddt] - np.diff(Ct_s[:,:,:ddt+1],axis=2))  # dX=dC/dt
    dw = (dec_dx[:,:,5] - Ct_s[:,:,5])  #dX=C
    J_ = dw/1 + J_/1  #single step
    #J_ = np.mean(dw,axis=2)/1 + J_/1  #time window
    
    plt.figure()
    plt.plot(dec_dx.reshape(N**2,win).T,'-o')
    
# %% testing
#kbT, w_rand, w_strc, P, mode = 1, 1, 0., 5, 'random'
pars = kbT, w_rand, w_strc, P, mode
St, _, _ = NeuralDynamics(N,T,pars,J_)

plt.figure()
plt.imshow(St,aspect='auto')
match = [np.abs(np.dot(M[ii],St[:,-1])) for ii in range(P)]
plt.figure()
plt.plot(match)

# %%
###############################################################################
# %% Statistical analysis
###############################################################################
# %%
def neglog(J,x,lamb):
    N = len(x)
    pis = np.zeros(N)
    mus = np.log(pis/(1-pis))
    phi = -y*mus + np.log(1+np.exp(mus))
    Phi = 1/N*np.sum(phi)
    reg = np.linalg.norm(J)
    nll = Phi + lamb*reg
    return nll

def Xcorr(x, y): 
    "Plot cross-correlation (full) between two signals."
    N = max(len(x), len(y)) 
    n = min(len(x), len(y)) 
    if N == len(y): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    c = correlate(x / np.std(x), y / np.std(y), 'full') /n
    return c, lags

def spin_sta(xi, xj, D):
    T = len(xi)
    X = sp.linalg.hankel(np.append(np.zeros(D-2),xj[:T-D+2]),xj[T-D+1:])
    sta = np.linalg.pinv(X.T @ X) @ X.T @ xi
    return sta

# %%
ii,jj,win = 0,30, 100
xi, xj = St[ii,:], St[jj,:]
cc,ll = Xcorr(xi,xj)
mid = int(len(cc)/2)
pos = np.arange(mid-win,mid,1)
plt.figure()
plt.plot(ll[pos],cc[pos])
sta = spin_sta(xi,xj,win)
plt.figure()
plt.plot(ll[pos[:-1]],sta)
plt.figure()
plt.plot(dec_dx[ii,jj,:])
plt.plot(Ct_s[ii,jj,:]*10)

# %%
###############################################################################
# %% Biophysical model
###############################################################################
# %%
def G(s,m):
    K0 = 1
    H = 1
    K = K0*np.exp(2*m)
    gg = 1/(1+(s/K)**H)
    return gg


def dGdm(f,s,m,method='central',h=0.01):
    if method == 'central':
        return (f(s,m + h) - f(s,m - h))/(2*h)
    elif method == 'forward':
        return (f(s,m + h) - f(s,m))/h
    elif method == 'backward':
        return (f(s,m) - f(s,m - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
        
# %% dynamics
dt = 0.01
T = 100
time = np.arange(0,T,dt)
lt = len(time)
a = np.zeros(lt)
m = np.zeros(lt)
wa,wm = 1,10
Da,Dm = .1,.1
C = Dm*wa/(Da*wm)
a0 = 0.5
beta = 0  #0 for equ, 1 for NE
s = np.random.randn(lt)*1.+5
for tt in range(lt-1):
    a[tt+1] = a[tt] + dt*( -wa*(a[tt]-G(s[tt],m[tt]) ) ) + np.random.randn()*np.sqrt(dt*Da)
    m[tt+1] = m[tt] + dt*( -wm*(a[tt]-a0)*(beta-(1-beta)*C*dGdm(G,s[tt],m[tt])) ) \
    + np.random.randn()*np.sqrt(dt*Dm)
#

# %%
plt.figure()
plt.plot(time,s)
plt.plot(time,a,'--')
plt.plot(time,m,'-o')

# %% Minumum SAS model
###############################################################################
# %%
def FE_(a,m,L):
    fe = delta_m*(a-0.5)*(m0-m) + (a-0.5)*np.log((1+L/KI)/(1+L/KA))
    return fe
def FE(a,m,e):
    fe = np.abs(e-m)*(dm+np.abs(e-a)*dg)
    return fe
def dFdm_EP(f,a,m,L,method='central',h=0.01):
    if method == 'central':
        return (f(a,m + h,L) - f(a,m - h,L))/(2*h)
def dFdm_exact_(a,m,L):
    df = -m0*delta_m*(a-0.5)
    return df
def trans_a(f,a,m,L):
    a_ = 0 if a==1 else 1  #flipping
    p = np.exp(-(f(a_,m,L)-f(a,m,L))/kbT)
    if p>np.random.rand():
        aa = a_
    else:
        aa = a
    return aa
def trans_m(f,a,m,L):
    m_ = 0 if m==1 else 1
    p = np.exp(-(f(a,m_,L)-f(a,m,L))/kbT)
    if p>np.random.rand():
        mm = m_
    else:
        mm = m
    return mm
# %% parameters
delta_m, m0 = 2,1
dm, dg = 5,5
KI, KA = 18.2, 3000
kbT = 1.
ww,kk = 100,1
lt = 10000
Lt = np.linspace(0,1,lt)
# %% dynamics
rep = 100
a_, m_ = np.zeros((rep,lt)), np.zeros((rep,lt))
for rr in range(rep):
    at = np.zeros(lt)#np.random.randint(0,2,lt)
    mt = np.zeros(lt)#np.random.randint(0,2,lt)
    for tt in range(lt-1):
        at[tt+1] = trans_a(FE,at[tt],mt[tt],Lt[tt])
        if tt % ww == 0:
            mt[tt+1] = trans_m(FE,at[tt],mt[tt],Lt[tt])
        else:
            mt[tt+1] = mt[tt]
    a_[rr,:], m_[rr,:] = at, mt

# %%
plt.figure()
plt.plot(np.mean(a_,axis=0)) #semilogx
plt.plot(np.mean(m_,axis=0))
#plt.plot(Lt)
# %% EP test
def dFdm_exact(a,m,e):
    df = -(e-m)*(dm+np.abs(e-a)*dg)
    return df
def flip(df,m):
    m_ = 0 if m==1 else 1
    p = np.exp(df/beta)
    if p>np.random.rand():
        mm = m_
    else:
        mm = m
    return mm
beta = 10
rep = 100
a_, m_ = np.zeros((rep,lt)), np.zeros((rep,lt))
for rr in range(rep):
    at = np.zeros(lt)
    mt = np.zeros(lt)
    mm = 0
    for tt in range(lt-1):
        at[tt+1] = trans_a(FE,at[tt],mt[tt],Lt[tt])
        if tt % ww == 0:
            ### dm = F_e - F_0 ###
            if Lt[tt] is not mt[tt]:
                dm = -1/beta*(dFdm_exact(at[tt],mt[tt],Lt[tt]) - dFdm_exact(at[tt],mt[tt],0))
                mm = mm+dm
                mt[tt+1] = max(min(mm,1),0)
                #mt[tt+1] = flip( dFdm_exact(at[tt],mt[tt],Lt[tt]) - dFdm_exact(at[tt],mt[tt],0),mt[tt] )
        else:
            mt[tt+1] = mt[tt]
    a_[rr,:], m_[rr,:] = at, mt
            

