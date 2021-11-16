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
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

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

def spiking(x,dt):
    """
    Produce Poisson spiking given spike rate
    """
    # rectification
#    x[x<0] = 0
#    x = x*gain
#     soft rectification
    x = 1/gain*np.log(1+np.exp(gain*x))
    
    spike = np.random.poisson(x*dt)
    return spike

def NL(x,spkM):
    """
    Passing x through logistic nonlinearity with spkM maximum
    """
    nl = spkM/(1+np.exp(-x))
#    nl = x
#    nl = np.tanh(x)
    return nl

# %% setup
#size and length
N = 300
T = 200
dt = 0.1
simtime = np.arange(0,T,dt)
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = 1.
E = (2*np.random.rand(N,1)-1)*Q
alpha = 1.  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M_ = np.random.randn(N,N)*g*scale
sparse = np.random.rand(N,N)
mask_J = np.random.rand(N,N)
mask_J[sparse>p] = 0
mask_J[sparse<=p] = 1
gain = 1.  #for lower firing rate
tau_r = 100*np.random.rand(N)  #important long self-spike time constant
spkM = .01  #if there's sigmoid nonlinearity

for ii in range(N) :
    jj = np.where(np.abs(M_[ii,:])>0)
    M_[ii,jj] = M_[ii,jj] - np.sum(M_[ii,jj])/len(jj)
M_ = M_ * mask_J

#coupling kernels
pad = 100
nbasis = 5
p_glm = 0.2
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T
allK = np.zeros((N,N,pad))  #number of kernels x length of time window
sparse = np.random.rand(N,N)
mask = np.random.rand(N,N)
mask[sparse>p_glm] = 0
mask[sparse<=p_glm] = 1
thetas = np.random.randn(N,N,nbasis)
for ii in range(N):
    for jj in range(N):
        temp = np.dot(thetas[ii,jj,:], Ks)
        if ii==jj:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = 0#temp
        else:
#            temp = np.dot( np.array([-1,0.5,0.2,-0.1,0.1]) , Ks )*np.random.choice([1,-1],1)[0]
            allK[ii,jj,:] = temp*mask[ii,jj] #0
            
#input parameters
wo = np.ones((N,1))
dw = np.zeros((N,1))
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
ft = np.concatenate((np.zeros(pad),ft))

#initial conditions
wo_len = np.zeros(simtime_len)
zt = np.zeros(simtime_len)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,simtime_len))
rt = np.random.randn(N,simtime_len)
spks = np.zeros((N,simtime_len))

xt[:,0] = x0
z = z0

# %% learning
P = (1.0/alpha)*np.eye(N)
for tt in range(pad+1, len(simtime)):
    ### GLM-RNN
    lin_v = (1-dt/tau_r)*rt[:,tt-1] + spks[:,tt-1]*dt/tau_r  #slow recover
#    lin_v = NL(np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]) , spkM)  #linear or nonlinear
    #    lin_v[lin_v<0] = 0  #rectification
    rt[:,tt] = lin_v
    spk = spiking( (M_ @ lin_v) , dt)  #generate spike s with current u
#    spk[spk>0] = 1   #force binary
    spks[:,tt] = spk
    
    ###
    ### self-linear, coupling-kernel, GLM network
#    self_v = (1-dt/tau_r)*rt[:,tt-1] + spks[:,tt-1]*dt/tau_r  #slow recover
#    np.fill_diagonal(M_, 0)
#    coup_v = np.einsum('ijk,jk->i', (M_[:,:,None]*allK), (spks[:,tt-pad-1:tt-1]))  #coupling term
#    lin_v = self_v*0 + NL( self_v+coup_v , spkM)  #linear summation or nonlinear synaptic input
##    coup_v = NL( allK*(spks[:,tt-pad-1:tt-1])[None,:,:] , spkM )  #LN-synaptic coupling
##    coup_v = coup_v.sum(2)
#    lin_v = (M_ * coup_v).sum(0) + self_v
#    rt[:,tt] = lin_v  #subthrehold
#    spk = spiking( rt[:,tt], dt)  #Poisson nonlinearity and spiking
##    spk[spk>0] = 1
#    spks[:,tt] = spk 
    ###
    
    #reconstruct dynamics
    z = wo.T @ lin_v #tens
    
    #learning
    if np.mod(tt, learn_every) == 0:
        dr = (lin_v)  #taking derivative over the nonlinearity #deNL
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
plt.plot(ft)
plt.plot(zt,'--')
plt.subplot(212)
plt.plot(wo_len)

plt.figure()
plt.imshow(spks,aspect='auto')

# %% testing
zpt = np.zeros(len(simtime))

for tt in range(pad+1, len(simtime)):
    #GLM-RNN
    lin_v = (1-dt/tau_r)*rt[:,tt-1] + spks[:,tt-1]*dt/tau_r  #slow recover  #linear slow dynamics
#    lin_v = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM)  #linear or nonlinear here
    #lin_v[lin_v<0] = 0  #rectification
    rt[:,tt] = lin_v
    spk = spiking( (M_ @ lin_v) , dt)  #generate spike s with current u
#    spk[spk>0] = 1   #force binary
    spks[:,tt] = spk
    
    ###
    #self-linear, coupling-kernel, GLM network
#    self_v = (1-dt/tau_r)*rt[:,tt-1] + spks[:,tt-1]*dt/tau_r  #slow recover
#    coup_v = np.einsum('ijk,jk->i', (M_[:,:,None]*allK), (spks[:,tt-pad-1:tt-1]))  #coupling term
#    lin_v = self_v*0 + NL( self_v+coup_v , spkM)
#    rt[:,tt] = lin_v  #subthrehold
#    spk = spiking( lin_v, dt)  #Poisson nonlinearity and spiking
##    spk[spk>0] = 1
#    spks[:,tt] = spk 
    
    #reconstruct dynamics
    z = wo.T @ lin_v#rt[:,tt]
    zpt[tt] = z

plt.figure()
plt.plot(ft)
plt.plot(zpt*1)

plt.figure()
plt.imshow(spks, aspect='auto')
plt.figure()
plt.plot(spks.T)

# %% Autonomous dynamics
rt_p = np.zeros_like(rt)
spks_p = np.zeros_like(spks)
signal = np.sin(simtime/20)
stim = np.random.randn(N, simtime_len)*30#signal*np.ones((N,simtime_len))*0

for tt in range(1, len(simtime)):
    #GLM-RNN
#    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM)  #linear here
    rec_pot = (1-dt/tau_r)*rt_p[:,tt-1] + spks_p[:,tt-1]*dt/tau_r  #slow recover
#    rec_pot[rec_pot<0] = 0  #rectification
    rt_p[:,tt] = rec_pot
    spk = spiking( (M_ @ rt_p[:,tt]) + stim[:,tt] , dt)  #generate spike s with current u
#    spk[spk>0] = 1
    spks_p[:,tt] = spk
    #reconstruct dynamics
    z = wo.T @ rec_pot#rt[:,tt]
    zpt[tt] = z

plt.figure()
plt.plot(ft)
plt.plot(zpt*1)

plt.figure()
plt.imshow(spks_p, aspect='auto')
plt.figure()
plt.plot(rt_p.T)

# %%
#linear estimate
#W_ = np.linalg.pinv(rt_p @ rt_p.T) @ rt_p @ spks_p.T
W_ = np.linalg.pinv(spks_p @ spks_p.T) @ spks_p @ rt_p.T
plt.figure()
plt.plot(M_.reshape(-1), W_.reshape(-1), 'o')
# %%
Cov_spk = np.cov(spks_p)
plt.figure()
plt.plot(M_.reshape(-1), Cov_spk.reshape(-1), 'o')
# %%
plt.figure()
G = np.linalg.pinv(np.eye(N)-M_)
plt.plot(G.reshape(-1), Cov_spk.reshape(-1), 'o')
# %%
plt.figure()
cut = 3
uu,ss,vv = np.linalg.svd(np.cov(spks_p))
lrM = uu[:,:cut] @ vv[:cut,:]
plt.plot(M_.reshape(-1), lrM.reshape(-1), 'o')

###############################################################################
# %% Comparison of parameters
###############################################################################
# %%
print('MSE_train:', sum((zt-ft[:])**2)/sum(zt**2))
print('MSE_test:', sum((zpt-ft[:])**2)/sum(zt**2))

# %% 
# G-FORCE, linear f, coupling k, low-rank, deterministic
params = np.array(['G-FORCE', r'linear $f$', r'coupling $k$', 'binary', r'homogeneous $h$',\
          'deterministic','fast coupling $k$', 'spike-FORCE'])
train = np.array([0.1, 0.39, 0.16, 0.11, 0.19, 0.09, 0.09, 0.108])
test = np.array([0.37, 2.44, 1.9, 1.26, 2.1, 0.25, 0.19, 0.608])
order = np.array([0,7,6,5,3,2,4,1])
order = np.array([0,])

# %%
params = np.array(['G-FORCE', 'spike-FORCE', r'$g=0.15$', r'$g=15$', r'identical $h$', r'linear $f$',])
train = np.array([0.1, 0.019, 0.82, 0.094, 0.19, 0.39])
test = np.array([0.19, 0.608, 0.99, 0.57, 2.1, 2.44])
order = np.array([0,1,2,3,4,5])

# %%
x = np.arange(len(order))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train[order], width, label='train')
rects2 = ax.bar(x + width/2, test[order], width, label='test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE',fontsize=40)
ax.set_xticks(x)
ax.set_xticklabels(params[order], rotation=45)
ax.legend(fontsize=40)

fig.tight_layout()