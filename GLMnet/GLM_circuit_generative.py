# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 02:22:21 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 

#%matplotlib qt5

# %%
# GLM network model
###############################################################################
# %% basis function
def basis_function1(nkbins, nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
    dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
    width = 5.*dbcenter # width of each bump
    bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*0.5+0.5)  #raise-cosine function formula
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] #constructing bases
    return np.array(BBstm).T  #time x basis

plt.plot(basis_function1(150,6))

def flipkernel(k):
    """
    flipping kernel to resolve temporal direction
    """
    return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
def kernel(theta, pad):
    """
    Given theta weights and the time window for padding,
    return the kernel contructed with basis function
    """
    nb = len(theta)
    basis = basis_function1(pad, nb)  #construct basises
    k = np.dot(theta, basis.T)  #construct kernels with parameter-weighted sum
    return flipkernel(k)

# %% Poisson GLM
def NL(x):
    """
    Nonlinearity for Poisson GLM
    """
    return 1/(1+np.exp(-x))  #np.exp(x)  #

def Pois_spk(lamb, delt):
    """
    Poisson process for spiking
    """
    y= np.random.poisson(lamb*delt)
    return y

def coupled_GLM():
    spks = 0
    return spks

# %% network settings
def GLM_cir(ss,gg):
    nn = 3  #number of neurons in the circuit
    pad = 100  #time window used for history-dependence
    T = 10000  #time steps
    deltt = 0.01  #time bins
    spks = np.zeros((nn,T))  #recording all spikes
    Psks = np.zeros((nn,T))  #recording all spiking probability
    #GLM settings
    nb = 6  #number of basis function used to construct kernel
    ks = np.random.rand(nn,nb)  #stimulus filter for each neuron
    hs = np.random.randn(nn,nn,nb)  #coupling filter between each neuron and itself
    v1 = np.array([1,0,1])#np.random.randn(nneuron)
    v2 = np.array([0,0,1])
    v3 = np.array([1,1,0])
    ww = gg*(np.outer(v1,v1) + np.outer(v2,v2) + np.outer(v3,v3) + np.random.randn(nn,nn)*.1)
    np.fill_diagonal(ww,-ss*np.ones(nn))
    hs[:,:,1] = ww
    Ks = np.array([kernel(ks[kk,:],pad) for kk in range(0,nn)])
    Hs = np.array([kernel(hs[ii,jj,:],pad) for ii in range(0,nn) for jj in range(0,nn)]).reshape(nn,nn,pad)
    mus = np.random.randn(nn)*0.1  #fiting backgroun
    #stimulus (noise for now)
    It = np.random.randn(nn,T)*.5 + .0*np.repeat(np.sin(np.linspace(0,T,T)/200),nn).reshape(T,nn).T
    It = It*.05
    for tt in range(pad,T):
        Psks[:,tt] = NL(np.sum(Ks*It[:,tt-pad:tt],axis=1) + mus + \
            np.einsum('ijk,jk->j',  Hs, spks[:,tt-pad:tt]) )
        spks[:,tt] = Pois_spk(Psks[:,tt], deltt)
    return Psks, spks

# %%
Psks, spks = GLM_cir(10,1)
plt.figure()
plt.subplot(211)
plt.imshow(Psks, aspect='auto')
plt.subplot(212)
plt.imshow(spks, aspect='auto')

# %%
plt.figure()
Psks, spks = GLM_cir(-1,1)
plt.subplot(221)
plt.imshow(Psks, aspect='auto')
plt.subplot(222)
Psks, spks = GLM_cir(-1,10)
plt.imshow(Psks, aspect='auto')
plt.subplot(223)
Psks, spks = GLM_cir(-10,-1)
plt.imshow(Psks, aspect='auto')
plt.subplot(224)
Psks, spks = GLM_cir(-10,-10)
plt.imshow(Psks, aspect='auto')

# %%
plt.figure()
Psks, spks = GLM_cir(1,10)
plt.subplot(221)
plt.imshow(spks, aspect='auto')
plt.subplot(222)
Psks, spks = GLM_cir(1,-10)
plt.imshow(spks, aspect='auto')
plt.subplot(223)
Psks, spks = GLM_cir(10,10)
plt.imshow(spks, aspect='auto')
plt.subplot(224)
Psks, spks = GLM_cir(10,-10)
plt.imshow(spks, aspect='auto')

# %%
# Neural circuit determinisitic patterns
###############################################################################
# %% network settings
#time course
dt = 0.001  #ms
T = 3  #total time
time = np.arange(0,T,dt)   #time axis
lt = len(time)  #step

#network and stimuli parameters
N = 2  #number of neurons

### Bistable ###
Ithresh = np.array([5, 5]);
W = np.array([[1.1, -0.15],
              [-0.15, 1.1]])  #weight matrix
rinit1 = np.array([50, 55])
Iapp1 = np.array([ 0, 30])
Iapp2 = np.array([30, 0])
### Line attractor ###
#Ithresh = np.array([-20, -20]);
#W = np.array([[0.7, -0.3],
#              [-0.3, 0.7]])  #weight matrix
#rinit1 = np.array([30, 75])
#Iapp1 = np.array([ 1, 0])
#Iapp2 = np.array([2, 0])
### Oscillation attractor ###
Ithresh = np.array([8, 20]);
W = np.array([[2.2, -1.3],
              [1.2, -0.1]])  #weight matrix
rinit1 = np.array([80, 0])
Iapp1 = np.array([ 0, 0])
Iapp2 = np.array([-10, 0])

#initialization
r = np.zeros((N,lt))    # array of rate of each cell as a function of time
rmax = 100;             # maximum firing rate
tau = 0.01;             # base time constant for changes of rate

#stimuli
Iapp = np.zeros((N,lt))      # Array of time-dependent and unit-dependent current
Ion1 = 1;                    # Time to switch on
Ion2 = 2;                    # Time to switch on
Idur = 0.2;                  # Duration of current

non1 = round(Ion1/dt)            # Time step to switch on current
noff1 = round((Ion1+Idur)/dt)    # Time step to switch off current
non2 = round(Ion2/dt)            # Time step to switch on current
noff2 = round((Ion2+Idur)/dt)    # Time step to switch off current

Iapp[:,non1:noff1] = np.einsum('ij,jk->jk',Iapp1[None,:],np.ones((2,noff1-non1)))
Iapp[:,non2:noff2] = np.einsum('ij,jk->jk',Iapp2[None,:],np.ones((2,noff2-non2)))

r[:,0] = rinit1                # Initialize firing rate

# %% Dynamics of rate network
for tt in range(1,lt):
    I = W @ r[:,tt-1] + Iapp[:,tt-1]                                    # total current to each unit
    newr = r[:,tt-1] + dt/tau*(I-Ithresh-r[:,tt-1])                     # Euler-Mayamara update of rates
    r[:,tt] = np.amax(np.hstack((newr[:,None],np.zeros((N,1)))),axis=1) # rates are not negative
    r[:,tt] = np.amin(np.hstack((r[:,tt][:,None],rmax*np.ones((N,1)))),axis=1)
    
plt.figure()
plt.subplot(212)
plt.plot(time,Iapp.T)
plt.xlabel('time (s)',fontsize=30)
plt.ylabel('input',fontsize=30)
plt.subplot(211)
plt.plot(time,r.T)
plt.ylabel('spike rate',fontsize=30)

# %%spiking network
tau_m = 0.01  #membrane time constant  #0.01
tau_r = 0.05  #synaptic rise time scale  #0.05
tau_d = 0.01   #synaptic decay time  #0.01
v_the = 9.5    #spik threshold  #13  #7.5  #9.5
v_res = -1     #reset potential after spiking  #-10 #-1 #-1
lamb = 35    #factor from rate to spiking networks  #140  #70  #35
vm = np.zeros((N,lt))  #for membrane potential
rs = np.zeros((N,lt))  #for spike rate
ss = np.zeros((N,lt))  #for synaptic input
spk = np.zeros((N,lt)) #for spikes
for tt in range(1,lt):
    #neural dynamics
    vm[:,tt] = vm[:,tt-1] + dt*(1/tau_m)*(-vm[:,tt-1] + lamb*W @ rs[:,tt-1] + Iapp[:,tt-1]*5+10) #,embrane potential
    rs[:,tt] = rs[:,tt-1] + dt*(-rs[:,tt-1]/tau_d + ss[:,tt-1])  #spike rate
    ss[:,tt] = ss[:,tt-1] + dt*(-ss[:,tt-1]/tau_r + 1/(tau_d*tau_r)*spk[:,tt-1])  #synaptic input
    #spiking process
    poss = np.where(vm[:,tt]>v_the)[0]  #recording spikes
    if len(poss)>0:
        spk[poss,tt] = 1
    posr = np.where(spk[:,tt-1]>0)[0]  #if spiked
    Vmask = np.ones(N)
    if len(posr)>0:
        vm[posr,tt] = v_res  #reseting spiked neurons
        spk[posr,tt] = 0

plt.figure()
plt.subplot(311)
plt.plot(time,vm.T)
plt.ylabel('voltage',fontsize=30)
plt.subplot(312)
plt.plot(time,spk.T,'-o')
plt.ylabel('spikes',fontsize=30)
plt.subplot(313)
plt.plot(time,Iapp.T)
plt.xlabel('time (s)',fontsize=30)
plt.ylabel('input',fontsize=30)

# %%


# %%
# GLM inference test
###############################################################################
#from GLM_debug import build_matrix, build_convolved_matrix
# %% design matrix function
def build_matrix(stimulus, spikes, pad, couple):
    """
    Given time series stimulus (T time x N neurons) and spikes of the same dimension and pad length,
    build and return the design matrix with stimulus history, spike history od itself and other neurons
    """
    T, N = spikes.shape  #neurons and time
    SN = stimulus.shape[0]  #if neurons have different input (ignore this for now)
    
    # Extend Stim with a padding of zeros
    Stimpad = np.concatenate((stimulus,np.zeros((pad,N))),axis=0)
    # Broadcast a sampling matrix to sample Stim
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    X = np.squeeze(Stimpad[S])
    if couple==0:
        X = X.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
    elif couple==1:
        X_stim = np.concatenate((np.ones((T,1)), X),axis=1)  #for DC component that models baseline firing
    #    h = np.arange(1, 6)
    #    padding = np.zeros(h.shape[0] - 1, h.dtype)
    #    first_col = np.r_[h, padding]
    #    first_row = np.r_[h[0], padding]
    #    H = linalg.toeplitz(first_col, first_row)
        
        # Spiking history and coupling
        spkpad = np.concatenate((spikes,np.zeros((pad,1))),axis=0)
        # Broadcast a sampling matrix to sample Stim
        S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
        X_h = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
        # Concatenate the neuron's history with old design matrix
        X_s_h = X_stim.copy()
        for hh in range(0,N):
            X_s_h = np.concatenate((X_s_h,X_h[hh]),axis=1)
        X = X_s_h.copy()
#        #print(hh)
    
    return X

def build_convolved_matrix(stimulus, spikes, Ks, couple):
    """
    Given stimulus and spikes, construct design matrix with features being the value projected onto kernels in Ks
    stimulus: Tx1
    spikes: TxN
    Ks: kxpad (k kernels with time window pad)
    couple: binary option with (1) or without (0) coupling
    """
    T, N = spikes.shape
    k, pad = Ks.shape
    
    Stimpad = np.concatenate((stimulus,np.zeros((pad,1))),axis=0)
    S = np.arange(-pad+1,1,1)[np.newaxis,:] + np.arange(0,T,1)[:,np.newaxis]
    Xstim = np.squeeze(Stimpad[S])
    Xstim_proj = np.array([Xstim @ Ks[kk,:] for kk in range(k)]).T
    
    if couple==0:
        X = np.concatenate((np.ones((T,1)), Xstim_proj),axis=1)
    elif couple==1:
        spkpad = np.concatenate((spikes,np.zeros((pad,N))),axis=0)
        Xhist = [np.squeeze(spkpad[S,[i]]) for i in range(0,N)]
        Xhist_proj = [np.array([Xhist[nn] @ Ks[kk,:] for kk in range (k)]).T for nn in range(N)]
        
        X = Xstim_proj.copy()
        X = np.concatenate((np.ones((T,1)), X),axis=1)
        for hh in range(0,N):
            X = np.concatenate((X,Xhist_proj[hh]),axis=1)
    return X

# %% inference method (single)
nneuron = 1
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
Y = spk[nneuron,:].T  #spike train of interest
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
stimulus = Iapp[nneuron,:][:,None]  #same stimulus for all neurons
X = build_convolved_matrix(stimulus, spk.T, Ks, couple)  #design matrix with features projected onto basis functions
###pyGLMnet function with optimal parameters
glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-5, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)

# %% simulation 
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
plt.figure()
plt.plot(time,Y*1.,label='data')  #ground truth
plt.plot(time,yhat,'--',label='GLM')
plt.plot(time,1-np.exp(-10*yhat))
plt.xlabel('time (s)',fontsize=30)
plt.ylabel('activity',fontsize=30)
plt.legend(fontsize=25)

# %%
y_test_hat = glm.predict(X)
plt.plot(y_test_hat,'-o')

# %%
spk_GLM = np.random.binomial(1, y_test_hat)#(1,1-np.exp(-5*yhat))
plt.plot(Y)
plt.plot(spk_GLM,'-o')

# %%reconstruct kernel
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


