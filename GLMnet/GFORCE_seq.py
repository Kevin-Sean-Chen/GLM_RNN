# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:29:08 2021

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

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5

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
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
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

def deNL(x):
    de = spkM*x*1/np.cosh(x)**2  #derivative of tanh
    #de = x*2/(1+np.exp(-2*x))**2 * np.exp(-2*x)
    return de

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
#    x[x<0] = 0
    x = 1/gain*np.log(1+np.exp(gain*x))
    #x[x>100] = 100
    spike = np.random.poisson(x*dt)  #Poisson process
#    spike = x*dt #rate model
    return spike

# %% setup
#size and length
N = 500
T = 100
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
p_glm = .2
g = 15  # g greater than 1 leads to chaotic networks.
Q = 1.
E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
nbasis = 5
pad = 100
spkM = 1.
gain = 1.
thetas = np.random.randn(N,N,nbasis)/1  #tensor of kernel weights
M_ = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1

#for ii in range(N) :
#    jj = np.where(np.abs(M_[ii,:])>0)
#    M_[ii,jj] = M_[ii,jj] - np.sum(M_[ii,jj])/len(jj)
#M_ = M_ * mask

Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((N,N,pad))  #number of kernels x length of time window
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p_glm] = 0
mask[sparse<=p_glm] = 1
for ii in range(N):
    for jj in range(N):
        temp = np.dot(thetas[ii,jj,:], Ks)
        if ii==jj:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*1.
        else:
            #temp = temp - np.mean(temp)
            #temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = temp*mask[ii,jj]

#input parameters
pN = 100 ### length, number of neurons, in this sequence
wo = np.ones((N,1))
dw = np.zeros((N,1))
wf_w = 2.0*(np.random.randn(N,nbasis)-0.5)
simtime_len = len(simtime)
#def get_cov(size=20, length=50):    
#    x = np.arange(size)
#    cov = np.exp(-(1 / length) * (x - np.atleast_2d(x).T)**2)
#    return cov
#l = 10000
#Km = get_cov(simtime_len, l)  #sequence target pattern of population
def get_seq(N,T,width):
    Km = np.zeros((N,T))
    xx = np.arange(T)
    tile = int(T/N)
    ii = 0
    for nn in range(N):
        Km[nn,:] = np.exp(-(xx-ii)**2/width**2)
        ii+=tile
    return Km
l = 10
Km = get_seq(N,(simtime_len),l)
#Km = get_seq(N,int(len(simtime)/4),l)
ft = Km*1-0.  #rescaled target pattern
#ft = np.concatenate((ft,ft),1)
#ft = np.concatenate((ft,ft),1)
#ft = np.repeat(ft, 3, axis=1)  #three repeats for training
ft = np.concatenate((np.zeros((N,pad)), ft), axis=1)  #padding for data length
stim_t = np.convolve(np.random.randn(simtime_len),np.ones(50),'same')  #cue
stim_t = stim_t - min(stim_t)
stim_t = stim_t/max(stim_t)*.1  #rescaled input cue

#initial conditions
wo_len = np.zeros(simtime_len)
zt = np.zeros_like(ft)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn(N)
xt = np.zeros((N,pad))
rt = np.zeros((N,pad))
spks = np.zeros((N,simtime_len))

xt[:,0] = x0
rt[:,0] = NL(xt[:,0],spkM)
z = z0

# %% learning
P = (1.0/alpha)*np.eye(N)
for tt in range(pad+1, len(simtime)):
    #GLM-RNN
#    tens = NL( np.einsum('ijk,jk->i',  allK, spks), spkM) 
#    spks_ = spiking( (M_ @ tens +0) , dt) #+ stim[:,tt]
#    spks = np.concatenate((spks[:,1:],spks_[:,None]),axis=1)  #time iterative update
    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM) + stim_t[tt]
    spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
    spks[:,tt] = spks_
    rt = tens
    
    #reconstruct dynamics
    z = wo* tens[:,None]  #is now a vector
    
    #learning
    if np.mod(tt, learn_every) == 0:
        dr = (tens)  #taking derivative over the nonlinearity #deNL
        k = (P @ dr)[:,None]
        rPr = dr[:,None].T @ k
        c = 1.0/(1.0 + rPr)
        P = P - (k @ k.T) * c  #projection matrix
    	
        # update the error for the linear readout
        e = z-ft[:,tt][:,None] ### how is error computed!
	
    	# update the output weights
        dw = -(e*k*c)#[:,None]
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M_[:,:] = M_[:,:] + mask*np.repeat(dw,N,1)#dw
        #np.repeat(dw,N,1).T
#        M_ = M_*mask
     
    #print(tt,z)
    # Store the output of the system.
    zt[:,tt] = np.squeeze(z)
    wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))
    
# %% plotting
plt.figure()
#error_avg = sum(abs(zt-ft[:]))/simtime_len
#print(['Training MAE: ', str(error_avg)]) 
print('mean firing:', np.mean(spks))  
print(['Now testing... please wait.'])

test = zt
test[zt<0]=0
plt.subplot(311)
plt.imshow(ft,aspect='auto')
plt.subplot(312)
plt.imshow(test,aspect='auto')
plt.subplot(313)
plt.plot(wo_len)

plt.figure()
plt.imshow(spks,aspect='auto')

# %% testing
zpt = np.zeros_like(zt)
M0 = np.random.randn(N,N)*g*scale
spks = np.zeros((N,simtime_len))

for tt in range(pad+1, len(simtime)):
    #GLM-RNN
#    tens = NL( np.einsum('ijk,jk->i',  allK, spks), spkM)
#    spks_ = spiking( (M_ @ tens +0) , dt) #+ stim[:,tt]
#    spks = np.concatenate((spks[:,1:],spks_[:,None]),axis=1)  #time iterative update
    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM) + stim_t[tt]
    spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
    spks[:,tt] = spks_
  
    #reconstruct dynamics
    z = wo* tens[:,None]  #is now a vector
    zpt[:,tt] = np.squeeze(z)

plt.figure()
plt.subplot(211)
plt.imshow(ft,aspect='auto')
plt.subplot(212)
plt.imshow(zpt,aspect='auto')

plt.figure()
plt.imshow(spks, aspect='auto')
error_avg = sum(abs(zpt-ft[:]))/simtime_len
print(['Training MAE: ', str(error_avg)]) 
print('mean firing:', np.mean(spks))

# %%
temp = spks.copy()
locs = np.zeros(N)
for nn in range(N):
    locs[nn] = np.where(temp[nn,:]==np.max(temp[nn,:]))[0][0]
sort_seq = np.argsort(locs)
plt.figure()
plt.imshow(temp[sort_seq,:]/np.max(temp[sort_seq,:],1)[:,None],aspect='auto')
plt.xlabel('time', fontsize=40)
plt.ylabel('cells', fontsize=40)

# %% draw spiks
p_rate = temp[sort_seq,:]/np.max(temp[sort_seq,:],1)[:,None]
cutoff = 300
p_rate = p_rate[cutoff:,pad:]
spk_raster = np.zeros((p_rate.shape[0], p_rate.shape[1]))
spk_raster[p_rate*1.> np.random.rand(p_rate.shape[0], p_rate.shape[1])] = 1
plt.figure()
plt.imshow(spk_raster, aspect='auto')
plt.xlabel('time', fontsize=40)
plt.ylabel('cells', fontsize=40)

test = np.where(spk_raster>0)
plt.figure()
plt.plot(np.flipud(test[1][:,None]),np.flipud(test[0][:,None]),'ko')

# %% analysis
#plt.figure()
#plt.plot(Rs_tr,MSE_tr,'o',markersize=20,label='test')
#plt.plot(Rs_te,MSE_te,'o',markersize=20,label='train')
#plt.xlabel('mean Poisson rate',fontsize=40)
#plt.ylabel('MSE',fontsize=40)
#plt.legend(fontsize=30)

# %% Target dynamics
###############################################################################
# %% moving average
npulse = 10
imp_a = np.random.randn(npulse)*10
imp_dur = 30
tau = 5
time_points = np.random.choice(simtime_len,npulse)
temp = np.zeros(simtime_len)
for ii,tt in enumerate(time_points):
    temp[tt:tt+imp_dur] = imp_a[ii]
#input_ = np.array([temp[tt:tt+imp_dur] = imp_a[ii] for ii,tt in enumerate(time_points)])
input_ = temp
target = np.zeros(simtime_len)
cnt = 0
for tt in range(simtime_len-1):
    target[tt+1] = target[tt] + dt/tau*(input_[tt])

plt.figure()
plt.plot(input_)
plt.plot(target)

# %%
impulse = np.random.randint(-1,2,N)
stim_ = np.zeros((N,simtime_len))
for ii,tt in enumerate(time_points):
    stim_[:,tt:tt+imp_dur] = np.repeat(impulse[:,None]*imp_a[ii],imp_dur,axis=1)

# %% 3-bit (with higher dimension)
    
    
###############################################################################
# %% Back to determinisitic model now
###############################################################################
# %%
# %% parameters
N = 500  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.  #learning initial constant
dt = 0.1
nsecs = 200
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
#M[sparse>p] = 0
mask = np.random.rand(N,N)
mask[sparse>p] = 0
mask[sparse<=p] = 1
#M = M*mask  ### fully connected here!!

nRec2Out = N
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
pN = 50
def get_seq(N,T,width):
    Km = np.zeros((N,T))
    xx = np.arange(T)
    tile = int(T/N)
    ii = 0
    for nn in range(N):
        Km[nn,:] = np.exp(-(xx-ii)**2/width**2)
        ii+=tile
    return Km
l = 100
Km = get_seq(N,simtime_len,l)
ft = Km*1  #rescaled target pattern

wo_len = np.zeros((1,simtime_len))    
zt = np.zeros((N,simtime_len))
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.zeros((N,simtime_len))

x = x0
r = np.tanh(x)
z = z0

# %% FORCE learning
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
for t in range(len(simtime)-1):
    ti = ti+1
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]  #xt[:,t] = x[:,0]
    z = wo*r #wo.T @ r
    
    if np.mod(ti, learn_every) == 0:
        k = P @ r;
        rPr = r.T @ k
        c = 1.0/(1.0 + rPr)
        P = P - k @ (k.T * c)  #projection matrix
        
        # update the error for the linear readout
        e = z-ft[:,ti][:,None]
        
        # update the output weights
        dw = -e*k*c
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M[:,:] = M[:,:] + mask*np.repeat(dw,N,1)#0.0001*np.outer(wf,wo)
        #np.repeat(dw,N,1).T#.reshape(N,N).T
        #np.outer(wf,wo)
        #np.repeat(dw.T, N, 1);
#        M = M*mask           

    # Store the output of the system.
    zt[:,ti] = np.squeeze(z)
    wo_len[0,ti] = np.sqrt(wo.T @ wo)	

zt = np.squeeze(zt)
error_avg = sum(abs(zt-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.figure()
plt.plot(ft.T)
plt.plot(zt.T,'--')

plt.figure()
plt.imshow(rt,aspect='auto')
# %% testing
zpt = np.zeros((N,simtime_len))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)-1):
    ti = ti+1 
    
    x = (1.0-dt)*x + M @ (r*dt) #+ wf * (z*dt)
    r = np.tanh(x)
    z = wo*r#wo.T @ r

    zpt[:,ti] = z.squeeze()

zpt = np.squeeze(zpt)
plt.figure()
plt.subplot(211)
plt.imshow(ft,aspect='auto')
plt.subplot(212)
plt.imshow(zpt,aspect='auto')

# %%
locs = np.zeros(N)
for nn in range(N):
    locs[nn] = np.where(zpt[nn,:]==np.max(zpt[nn,:]))[0][0]
sort_seq = np.argsort(locs)
plt.figure()
plt.imshow(zpt[sort_seq,:],aspect='auto') #/np.max(zpt[sort_seq,:],1)[:,None]

