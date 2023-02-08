# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:23:32 2021

@author: kevin
"""

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
# %% functions
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

def Decision_targets(stim_t, mu, noise_s):
    prob_dm = max(stim_t)  #probability directly encoded in strength
    T = len(stim_t)
    output = np.zeros(T)+0
    for tt in range(1,T):
        if output[tt-1]>=100:
            output[tt] = 100
        elif output[tt-1]<0:
            output[tt] = 0
        else:
            if np.random.rand()<prob_dm:
                k = 1*mu
            else:
                k = -1*mu*0
            output[tt] = output[tt-1] + k*np.random.rand()*noise_s
    
    return output

# %%
lt = 200
mu_dm = 2
noise_dm = 1
stim_t = np.ones(lt)*0.3
#stim_t[pad*2:] = 0
output = Decision_targets(stim_t, mu_dm, noise_dm)
plt.figure()
plt.plot(output)

pc_s = np.array([0.1, 0.3, 0.5, 0.7, 0.9])  #probability of high or low bound
reps = 10  #repeat each probability

# %% setup
#size and length
N = 300
dt = 0.1
learn_every = 2  #effective learning rate

#network parameters
p = .2  #sparsity of connection
p_glm = 0.2
g = 1.5  # g greater than 1 leads to chaotic networks.
Q = .5
E = (2*np.random.rand(N,1)-1)*Q
alpha = .2  #learning initial constant
scale = 1.0/np.sqrt(p*N)  #scaling connectivity
nbasis = 5
pad = 50
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

#initial conditions
wo_len = np.zeros(lt)
zt = np.zeros(lt)
x0 = 0.5*np.random.randn(N)
z0 = 0.5*np.random.randn(1)
xt = np.zeros((N,lt))
rt = np.zeros((N,lt))
spks = np.zeros((N,lt))

xt[:,0] = x0
rt[:,0] = NL(xt[:,0],spkM)
z = z0

# %% learning
recordings = np.zeros((reps, len(pc_s), lt))  #repeat x probability x time
for rr in range(reps):
#    trials = np.random.permutation(len(pc_s))
    trials = np.arange(0,len(pc_s))
    for pp in range(len(pc_s)):
        stim_t = np.ones(lt)*pc_s[trials[pp]]
#        stim_t[int(pad*1.5):] = 0 #pulse
        ft = Decision_targets(stim_t, mu_dm, noise_dm)  #generate target
        
        print('repeat:',rr, 'probability:',pc_s[trials[pp]])
        
        P = (1.0/alpha)*np.eye(N)
        for tt in range(pad+1, lt):
            #GLM-RNN
            tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM) + stim_t[tt]
            spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
        #    spks_[spks_>0] = 1
            spks[:,tt] = spks_
            rt[:,tt] = tens
               
            #reconstruct dynamics
            z = wo.T @ tens
            
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
             
            # Store the output of the system.
            zt[tt] = np.squeeze(z)
            wo_len[tt] = np.nansum(np.sqrt(wo.T @ wo))
            
            ### recordings through trials
            recordings[rr,pp,:] = zt

# %% testing
pc_test = .5
stim_t = np.ones(lt)*pc_test
stim_t[int(pad*1.5):] = 0 #pulse
zpt = np.zeros(lt)

for tt in range(pad+1, lt):
    #GLM-RNN
    tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM) + stim_t[tt]
    spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
#    spks_[spks_>0] = 1
    spks[:,tt] = spks_
    rt[:,tt] = tens
       
    #reconstruct dynamics
    z = wo.T @ tens
    zpt[tt] = z

plt.figure()
plt.plot(zpt)
#plt.figure()
#plt.imshow(spks, aspect='auto')
#plt.figure()
#plt.plot(spks.T)
# %%
test_dm = np.zeros((reps, len(pc_s), lt))  #repeat x probability x time
for rr in range(reps):
    trials = np.arange(0,len(pc_s))
    #np.random.permutation(len(pc_s))
    for pp in range(len(pc_s)):
        stim_t = np.ones(lt)*pc_s[trials[pp]]
        stim_t[int(pad*1.5):] = 0 #pulse
        zpt = np.zeros(lt)
        
        print('repeat:',rr, 'probability:',pc_s[trials[pp]])
        
        for tt in range(pad+1, lt):
            #GLM-RNN
            tens = NL( np.einsum('ijk,jk->i',  allK, spks[:,tt-pad-1:tt-1]), spkM) + stim_t[tt]
            spks_ = spiking( (M_ @ tens) , dt)  #generate spike s with current u
        #    spks_[spks_>0] = 1
            spks[:,tt] = spks_
            rt[:,tt] = tens
               
            #reconstruct dynamics
            z = wo.T @ tens
            zpt[tt] = z
        
        test_dm[rr,pp,:] = zpt
# %% analysis
plt.figure()
plt.plot(test_dm[:,4,75:150].T,'r')
plt.plot(test_dm[:,2,75:150].T,'g')
plt.plot(test_dm[:,0,75:150].T,'b')
plt.xlabel('decision period',fontsize=40)
plt.ylabel('decision variable',fontsize=40)

