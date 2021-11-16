# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:42:12 2021

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
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

#%matplotlib qt5
# %% FORCE functions
def Lorenz_dt(xyz, dt):
    ### Lorenz attractor in 3-D with parameters from the paper
    x, y, z = xyz
    dx = (10*(y-x))*dt
    dy = (x*(28-z)-y)*dt
    dz = (x*y-(8/3)*z)*dt
    return dx,dy,dz

def Lorenz_model(T,dt):
    ### Samples with ran length T and time step dt
    lt = int(T/dt)
    Xt = np.zeros((lt,3))
    Xt[0,:] = np.random.randn(3)*10  #initial condition within the dynamic range
    for tt in range(lt-1):
        dx,dy,dz = Lorenz_dt(Xt[tt,:], dt)
        Xt[tt+1,:] = Xt[tt,:]+[dx,dy,dz]
    return Xt

def spiking(x,dt):

    x = 1/gain*np.log(1+np.exp(gain*x))
#    x[x<0] = 0
    spike = np.random.poisson(x*dt*gain)
    return spike

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
    nl = spkM/(1+np.exp(x))
#    nl = x
    #nl = np.tanh(x)
    return nl

def deNL(x):
    de = x*1/np.cosh(x)**2  #derivative of tanh
    #de = x*2/(1+np.exp(-2*x))**2 * np.exp(-2*x)
    return de

# %%
Lor_3d = Lorenz_model(15,0.005) #proper scale for Lorenz

# %% setup
#size and length
N = 500
T = 300
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
p_glm = 0.2
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = .5
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
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = temp*1.
        else:
            #temp = temp - np.mean(temp)
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = temp*mask[ii,jj]

#input parameters
wo = np.ones((N,1))
dw = np.zeros((N,1))
wf_w = 2.0*(np.random.randn(N,nbasis)-0.5)
simtime_len = len(simtime)
amp = 0.7;
freq = 1/60;
rescale = 4
ft = 1*(amp/1.0)*np.sin(1.0*np.pi*freq*simtime*rescale) + \
    1*(amp/2.0)*np.sin(2.0*np.pi*freq*simtime*rescale) + \
    1*(amp/6.0)*np.sin(3.0*np.pi*freq*simtime*rescale) + \
    0*(amp/3.0)*np.sin(4.0*np.pi*freq*simtime*rescale)
#ft[ft<0] = 0
ft = ft*100
ft = Lor_3d[:,2]*6
#ft = ft*5#/1.5
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
    spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
#    spks_[spks_>0] = 1
    spks[:,tt] = spks_
    rt[:,tt] = tens
       
    #reconstruct dynamics
    z = wo.T @ tens #[:,tt]
    
    #learning
    if np.mod(tt, learn_every) == 0:# and tt>1000 and tt<3000:
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
        M_ = M_ + np.repeat(dw,N,1).T  #(E @ dw.T) #
        #np.repeat(dw,N,1).T#0.0001*np.outer(wf,wo)
#        M_ = M_*mask_J
     
    #print(tt,z)
    # Store the output of the system.
    zt[tt] = np.squeeze(z)
    wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))

# %% FORCE-plots
###############################################################################
# %%
plt.figure()
plt.subplot(212)
plt.plot(np.diff(wo_len))
plt.axvline(x=1000, color='k', alpha=0.6,linewidth=10)
plt.axvline(x=3000, color='grey', alpha=0.6,linewidth=10)
plt.ylim([-0,1])

#plt.figure()
plt.subplot(211)
plt.plot(ft[:-pad],linewidth=8)
plt.plot(zt,'--',linewidth=8)
plt.axvline(x=1000, color='k', alpha=0.6,linewidth=10)
plt.axvline(x=3000, color='grey', alpha=0.6,linewidth=10)

# %%
samp = 10
n_spk = np.sum(spks[:,2000:],1)
w_spk = np.where(n_spk>0)[0]
selected = spks[w_spk[np.random.randint(len(w_spk),size=samp)],:]
plt.figure()
base=0
for ss in range(samp):
    plt.plot(selected[ss,:]+base)
    base += max(selected[ss,:])
plt.axvline(x=1000, color='k', alpha=0.6,linewidth=10)
plt.axvline(x=3000, color='grey', alpha=0.6,linewidth=10)
plt.yticks([])

# %%
Lor_rec[:,2] = zt
# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(Lor_3d[:,0]*6, Lor_3d[:,1]*6, Lor_3d[:,2]*6, 'k',linewidth=6)
ax.plot3D(Lor_rec[pad+10:,0], Lor_rec[pad+10:,1], Lor_rec[pad+10:,2], 'b')
ax.set_xlabel('x',fontsize=60)
ax.set_ylabel('y',fontsize=60)
ax.set_zlabel('z',fontsize=60)
