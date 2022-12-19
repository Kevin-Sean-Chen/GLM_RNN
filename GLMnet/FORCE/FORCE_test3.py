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
N = 100  #number of neurons
p = 1.  #sparsity of connection
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
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
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
        M = M*mask           

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

# %% FORCE with network constraints!
###############################################################################
###############################################################################
# %%
#cd C:\Users\kevin\OneDrive\Documents\github\Chemotaxis_Model
import csv
# %%
### loading connectome
with open("herm_full_edgelist.csv") as f:
    reader = csv.DictReader(f)
    data = [r for r in reader]
    
allns = []
for i in range(0,len(data)):
    temp = data[i]['Source']
    allns.append(temp.strip())
    #temp = data[i]['Target']
    #allns.append(temp)
data[0]

allns_a = np.array(allns)
nID = np.unique(allns_a)  #all unique neural IDs
ns = len(nID)
AA = np.zeros((ns,ns))

for i in range(0,len(data)):
    typp = data[i]['Type']
    if typp=='chemical': #'electrical' or typp=='chemical':#
        so = data[i]['Source'].strip()
        ta = data[i]['Target'].strip()
        we = data[i]['Weight'].strip()
        #if so=='AWCR':                  #for picking!
        #    print(np.where(so==nID)[0])
        xx = np.where(so==nID)[0]
        yy = np.where(ta==nID)[0]
        if len(yy)==0:
            break
        AA[xx[0],yy[0]] = float(we)

# %%
# %% parameters
AA = AA[:300,:300]
N = AA.shape[0]  #number of neurons
p = .1  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.  #learning initial constant
dt = 0.1
nsecs = 500
learn_every = 2  #effective learning rate

scale = 1.0/np.sqrt(p*N)  #scaling connectivity
M = (AA + 0*np.random.randn(N,N))*g*scale
mask = np.random.rand(N,N)
mask[AA>0] = 1
#mask[AA==0] = 0

nRec2Out = N
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
wf = 2.0*(np.random.rand(N,1)-0.5)

simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

###target pattern
amp = 0.7;
freq = 1/60;
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) + \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) + \
    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
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
        M = M + np.repeat(dw,N,1).T
        M = M*mask           

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
x = x0
r = np.tanh(x)
z = z0
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

# %% full-FORCE required!!!