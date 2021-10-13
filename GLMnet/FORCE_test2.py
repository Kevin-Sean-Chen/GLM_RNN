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

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5
# %% parameters
N = 300  #number of neurons
p = .2  #sparsity of connection
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
rescale = 2
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
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
#        M = M*mask           

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
    nl = spkM/(1+np.exp(-2*x))
#    nl = x
    #nl = np.tanh(x)
    return nl

def deNL(x):
    de = x*1/np.cosh(x)**2  #derivative of tanh
    #de = x*2/(1+np.exp(-2*x))**2 * np.exp(-2*x)
    return de

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
#    N = len(x)
#    spike = np.random.rand(N) < x*dt*0.1
    x[x<0] = 0
    #x[x>100] = 100
    spike = np.random.poisson(x*dt)
#    spike = x*dt
    return spike

# %% setup
#size and length
N = 300
T = 500
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
p_glm = 0.2
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = 1.
E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
nbasis = 5
pad = 50
spkM = 1.
tau = 1
thetas = np.random.randn(N,N,nbasis)/1  #tensor of kernel weights
M_ = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask_J = np.random.rand(N,N)
mask_J[sparse>p] = 0
mask_J[sparse<=p] = 1

for ii in range(N) :
    jj = np.where(np.abs(M_[ii,:])>0)
    M_[ii,jj] = M_[ii,jj] - np.sum(M_[ii,jj])/len(jj)
M_ = M_ * mask_J

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
            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )
            allK[ii,jj,:] = temp*10.
        else:
            #temp = temp - np.mean(temp)
            #temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = temp*mask[ii,jj]

#input parameters
wo = np.ones((N,1))
dw = np.zeros((N,1))
wf_w = 2.0*(np.random.randn(N,nbasis)-0.5)
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 2
ft = 1*(amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    1*(amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    1*(amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft[ft<0] = 0
ft = ft*100#/1.5
#ft = Xt[:,0]*5  #test for chaos
ft = np.concatenate((np.zeros(pad),ft))

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

# %% learning
P = (1.0/alpha)*np.eye(N)
for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM)
    #xt[:,tt] = M_ @ tens #(1.0-dt)*xt[:,tt-1] + dt*( M_ @ tens )  #dt*( M_ @ tens ) #
    #spks[:,tt] = spiking( (xt[:,tt]) , dt)
    spks[:,tt] = spiking( (M_ @ tens) , dt)  #generate spike s with current u
    rt[:,tt] = tens
    #rt[:,tt] = rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]/tau
    
    #NL( (xt[:,tt]), spkM)   ### GLM form
    #rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]  ### ODE form
    #(1.0-dt)*rt[:,tt-1]/tau + dt*spks[:,tt]   #+ np.random.randint(0,5,N)  ### weird form
    #xt[:,tt] = ( np.einsum('ijk,ik->i',  (thetas @ Ks), rt[:,tt-pad:tt]) )
    #rt[:,tt] = spiking(NL(xt[:,tt],spkM) , dt) #NL( spiking(xt[:,tt], dt), spkM) #+ np.random.randint(0,5,N)
    
    #reconstruct dynamics
    z = wo.T @ tens #[:,tt]
    
    #learning
    if np.mod(tt, learn_every) == 0:
        dr = (tens)  #taking derivative over the nonlinearity #deNL
        k = (P @ dr)[:,None]
        rPr = dr[:,None].T @ k
        c = 1.0/(1.0 + rPr)
        P = P - (k @ k.T) * c  #projection matrix
    	
        # update the error for the linear readout
        e = z-ft[tt] ### how is error computed!
	
    	# update the output weights
        dw = -(e[:,None]*k*c)#[:,None]
        wo = wo + dw
        
        # update the internal weight matrix using the output's error
        M_ = M_ + np.repeat(dw,N,1).T  #(E @ dw.T)
        #np.repeat(dw,N,1).T#0.0001*np.outer(wf,wo)
#        M_ = M_*mask_J
     
    #print(tt,z)
    # Store the output of the system.
    zt[tt] = np.squeeze(z)
    wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))
    
# %% plotting
plt.figure()
error_avg = sum(abs(zt-ft[:-pad]))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.subplot(211)
plt.plot(ft[:-pad])
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
M0 = np.random.randn(N,N)*g*scale
#xt[:,:pad], spks[:,:pad], rt[:,:pad] = xt[:,-pad:], spks[:,-pad:], rt[:,-pad:]

for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM)
    #xt[:,tt] = (1.0-dt)*xt[:,tt-1] + dt*( M_ @ tens )  #dt*( M_ @ tens ) #
    #spks[:,tt] = spiking( (xt[:,tt]) , dt)
    spks[:,tt] = spiking( (M_ @ tens) , dt)
    rt[:,tt] = tens
    #rt[:,tt] = rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]/tau
    #NL( (xt[:,tt]), spkM)   ### GLM form
    #rt[:,tt-1] - dt*rt[:,tt-1]/tau + dt*spks[:,tt]  ### ODE form
    #(1.0-dt)*rt[:,tt-1]/tau + dt*spks[:,tt]   #+ np.random.randint(0,5,N)  ### weird form
    #xt[:,tt] = ( np.einsum('ijk,ik->i',  (thetas @ Ks), rt[:,tt-pad:tt]) )
    #rt[:,tt] = spiking(NL(xt[:,tt],spkM) , dt) #NL( spiking(xt[:,tt], dt), spkM) #+ np.random.randint(0,5,N)
    
    #reconstruct dynamics
    z = wo.T @ tens#rt[:,tt]
    zpt[tt] = z

plt.figure()
plt.plot(ft)
plt.plot(zpt*1)

plt.figure()
plt.imshow(spks, aspect='auto')

# %% Analysis
weights = M_.reshape(-1)
feature = np.zeros((N,N))
for ii in range(N):
    for jj in range(N):
        feature[ii,jj] = allK[ii,jj,:10].sum()
        #(allK[ii,jj,:]**2).sum()  #
        #thetas[ii,jj,0]

plt.figure()
plt.plot(feature.reshape(-1), weights.reshape(-1), 'o')
plt.xlabel('kernel feature',fontsize=30)
plt.ylabel('weight',fontsize=30)

# %%
xs,ys = np.where(M_<-50)
ks = allK[xs,ys,:]
kmea = ks.mean(axis=0)
kstd = ks.std(axis=0)
plt.figure()
plt.plot(kmea)
plt.fill_between(np.arange(len(kmea)),kmea-kstd, kmea+kstd, alpha=0.5)
plt.figure()
plt.plot(ks.T)


