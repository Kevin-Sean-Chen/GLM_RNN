# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 02:05:30 2022

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 00:00:21 2021

@author: kevin
"""


import numpy as np
import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import grad, jacobian

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

#%matplotlib qt5
# %%
def R_FORCEDistribution(N, g, index):
    radius = np.array([0.9, 0.7, 0.72, 1.2]) * g
    num_circles = radius.size
    partition_out = 0.01 * np.ones([1,1])
    partition_in = 1 - partition_out
    distance_thresold = 1.15
    if radius[num_circles-1] > 1.55:
        per_out = partition_out * N/2
        per_in = np.zeros([1, num_circles - 1])
        radius_in = radius[:-1]
        distance = np.absolute(radius_in - distance_thresold)
        percentage = (1 / distance) **(2 * g)
        percentage_normalized = percentage / np.sum(percentage)
        for i in range(num_circles - 2):
            a = np.maximum(np.floor(percentage_normalized[i] * N/2), 0.1 * N/2)
            per_in[0][i] = a
        per_in[0][-1] = partition_in * N/2 - np.sum(per_in)        
        per = np.concatenate((per_in, per_out), axis=1)
    else:
        radius_in = radius
        distance = np.absolute(radius_in - distance_thresold)
        percentage = np.tanh(-2 * distance) + 1
        percentage_normalized = percentage / np.sum(percentage)
        for i in range(num_circles -1):
            per[0,i] = np.floor(percentage_normalized[i] * N/2)
        
        per[0,-1] = N/2 - np.sum(per)   
    count = 0
    theta = np.zeros([1, int(N/2)])
    real_part = np.zeros([1, int(N/2)])
    imag_part_pos = np.zeros([1, int(N/2)])

    if (g < 1.8):
      theta_range = [0.02, np.pi/3, np.pi / 3, 2 * np.pi/3, 2 * np.pi/3, np.pi - 0.02]
    else:
      theta_range = [4 * np.pi/5, np.pi - 0.02, 2 * np.pi/5, 4 * np.pi/5, 0.02, 2 * np.pi/5]
    for i  in range(num_circles - 1):

      theta[0,count: count + int(per[0,i])] = np.linspace(theta_range[2 * i],  theta_range[2*i + 1] , num = int(per[0][i]))

      real_part[0,count: count + int(per[0,i])]= radius[i] * np.cos(theta[0,count: count + int(per[0,i])]);
      imag_part_pos[0,count: count + int(per[0,i])]= radius[i] * np.sin(theta[0,count: count + int(per[0,i])]);
      count = count + int(per[0][i])
    
    i = num_circles - 1
    if (g < 1.4):
      theta[0,count: count + int(per[0,i])] = np.linspace(0,  1.2 * np.pi /(g**2 * (num_circles - 1)), num = int(per[0,i]))
    else:
      theta[0,count: count + int(per[0,i])] = np.linspace(0.4 * np.pi, 0.6 * np.pi, num = int(per[0,i]))

    real_part[0,count: count + int(per[0,i])]= radius[i] * np.cos(theta[0,count: count + int(per[0,i])]);
    imag_part_pos[0,count: count + int(per[0,i])]= radius[i] * np.sin(theta[0,count: count + int(per[0,i])]);

    count = count + int(per[0][i])

    imag_part_neg = - imag_part_pos;
    
    # Generate the eigenvectors matrix 
    Temp = np.random.normal(0,1,[N,N])

    Temp = Temp - np.transpose(Temp)
    
    [D,V] = np.linalg.eig(Temp)
    
        
    eig_val = np.zeros([N]) * 1j

    M = np.zeros([N,N]) * 1j
    

    for i in range(int(N/2)):
        eig_val[2*i] = real_part[0,i] + 1j * imag_part_pos[0,i]
        eig_val[2*i + 1] = real_part[0,i] + 1j * imag_part_neg[0,i]
        
    M = np.dot(np.dot(V, np.diag(eig_val)), np.linalg.inv(V))
    M = np.real(M)
    per = per / (N/2)
    return M, per,radius, theta
# %%
def network(N, p, g):
    scale = 1.0/np.sqrt(p*N)  #scaling connectivity
    M = np.random.randn(N,N)*g*scale
    sparse = np.random.rand(N,N)
    mask = np.random.rand(N,N)
    mask[sparse>p] = 0
    mask[sparse<=p] = 1
    M = M*mask
    return M

def stim_ssm():
    amp = 1
    freq = 1/20;
    f_s = 1/50
    ft_ = amp*np.sin(np.pi*f_s*simtime)
    high_f = amp*np.sin(np.pi*freq*simtime*4)
    low_f = amp*np.cos(np.pi*freq*simtime*2)
    offset = .2
    pos_h = np.where(ft_>offset)[0]
    pos_l = np.where(ft_<=offset)[0]
    ft = np.zeros((2,simtime_len))
    ft[0,pos_h] = high_f[pos_h]*.5
    ft[1,pos_l] = low_f[pos_l]*2
    #ft = ft -np.mean(ft,1)[:,None]#+ 0.
    zt = np.zeros((2,simtime_len))
    zt[0,pos_h] = 1
    zt[1,pos_l] = 1
    zt = 2*zt[0,:]-1.#ft_.copy() ##for one D
    return ft.sum(0), zt

def RLS(P,r):
    k = P @ r
    rPr = r.T @ k
    c = 1.0/(1. + rPr)
    P = 1/1*(P - k @ (k.T * c))  #projection matrix
    return P, k, c

def Hebb(r1,r2,theta,q):
    r11,r22 = r1,r2
    r11[r1>theta] = q
    r11[r1<=theta] = -(1-q)
    r22[r2>theta] = q
    r22[r2<=theta] = -(1-q)
    return r11 @ r22.T

# %% parameters
### network
N = 300  #number of neurons
p = .2  #sparsity of connection
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.1  #learning initial constant
dt = 0.1
nsecs = 400
learn_every = 2  #effective learning rate
simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

M = network(N, p, g)
M_d = network(N, p, g)

nRec2Out = N
uu,ss,vv = np.linalg.svd(np.random.randn(N,N))
wo = np.zeros((nRec2Out,1))
dw = np.zeros((nRec2Out,1))
#wf = 2.0*(np.random.rand(N,1)-0.5)*.1
wf = uu[:,0][:,None]*1

wo_d = np.random.randn(nRec2Out,1)*.1#np.zeros((nRec2Out,2))
dw_d = np.zeros((nRec2Out,1))
wf_d = wf.copy()#
#wf_d = 2.0*(np.random.rand(N,1)-0.5)*.1
wf_d = uu[:,1][:,None]*1#np.repeat(wf,2,axis=1) #
#M_d += wf_d @ np.random.randn(nRec2Out,2).T#wo_d.T

w_h = 2.0*(np.random.rand(N,1)-0.5)*.1

###target pattern
ft, st = stim_ssm()
plt.figure()
plt.subplot(211)
plt.plot(ft.T)  #raw trace target  

plt.subplot(212)
plt.plot(st.T)  #underlying states

### initialization   
x0 = 0.5*np.random.randn(N,1)
z0 = 0.5*np.random.randn(1)
rt = np.zeros((N,simtime_len))
rt_d = rt*1

x = x0
r = np.tanh(x)
z = z0
x_d = x0
r_d = np.tanh(x_d)
z_d = np.zeros((1,1))

M, per,radius, theta = R_FORCEDistribution(N,g,0)
M_d = M*1

# %% FORCE learning
z_ft = np.zeros((1,simtime_len))
z_st = np.zeros((2,simtime_len))
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(nRec2Out)
P_d = P*1
proj = np.eye(N)*1
for t in range(len(simtime)-1):
    ti = ti+1
#    stt = np.array([st[ti]])[:,None]*.2
    ### observation network
    noise = np.random.randn(N,1)*0.0
#    hint = st[1,ti]-st[1,ti-1]#0
    x = (1.0-dt)*x + M @ (r*dt) + noise #+ wf_d @ stt #+ w_h*hint#+ proj @ (r_d*dt)
    r = np.tanh(x)
    rt[:,t] = r[:,0]
    z = wo.T @ r  # linear decoding
    inpt = np.array([ft[ti]])[:,None]  #either use this or z for driving
    
    ### latent state network
    x_d = (1.0-dt)*x_d + M_d @ (r_d*dt) + wf @ inpt + wf_d @ z_d + noise #w_h*hint#st[:,ti][:,None]#
    r_d = np.tanh(x_d)
    rt_d[:,t] = r_d[:,0]
    z_d = wo_d.T @ r_d  #state-reconstruction
    
    if np.mod(ti, learn_every) == 0:
        ### RLS calculation
        P, k, c = RLS(P, r)
        P_d, k_d, c_d = RLS(P_d, r_d)
        
        ### error terms
        e = z - ft[ti]
        e_d = z_d - ( 1.*st[ti]-0.) #- z
        e_J = M @ r - M_d @ r_d - wf @ inpt #- wf_d @ stt#wo.T @ r #- z_d #*st[0,ti]#w_h*hint*0
        
        ### update weights
        dw = -k*c*e.T
        wo = wo + dw  # target readout
        dJ = -e_J @ (P @ r).T
        M = M + dJ    # target network
#        M = M + wf @ dw.T
#        theta,q = .0,0.5
#        M = M + dt*(-M/100 + Hebb(r_d,r_d,theta,q)*.01/N + \
#                Hebb(rt[:,ti-1][:,None],r_d,theta,q)*.1/N)
        
#        Pt = np.array([np.exp(dw_d[:,0]@r_d),np.exp(dw_d[:,1]@r_d)])
#        Pt = np.array([np.exp(r_d.T @(np.outer(wf_d[:,0],dw_d[:,0])) @ r_d), \
#                              np.exp(r_d.T @ (np.outer(wf_d[:,1],dw_d[:,1])) @ r_d)])[:,:,0]
#        Pt /= sum(Pt)
#        dw_d =  -(k_d*c_d*(r_d*wf_d) @((1-Pt)).T)#*st[:,ti]*1#-k_d*c_d*(e_d*st[:,ti][:,None]).T #
#        dw_d = -P_d @ (r_d.T @ wf_d * r_d)*((1-Pt)).T*st[:,ti]
#        dw_d = -P_d @ (r_d) @ ((1-Pt)).T*st[:,ti]
#        dw_d = -k_d*c_d*(1*e_d).T*1  -((P_d @ (r_d))*((1-Pt)).T)#*st[:,ti]  ### modify this as transition!
        dw_d = -k_d*c_d*((1*e_d).T)
        wo_d = wo_d + dw_d
        M_d = M_d + wf_d @ dw_d.T  # state-network
#        theta,q = .0,0.5
#        M_d = M_d + dt*(-M_d/100 + Hebb(r_d,r_d,theta,q)*.05/N + \
#                        Hebb(rt_d[:,ti-1][:,None],r_d,theta,q)*.1/N)

    # Store the output of the system.
    z_ft[:,ti] = np.squeeze(z)
    z_st[:,ti] = z_d.squeeze()

z_ft = np.squeeze(z_ft)
error_avg = sum(abs(z_ft-ft))/simtime_len
print(['Training MAE: ', str(error_avg)])   
print(['Now testing... please wait.'])

plt.plot(z_ft.T)
plt.plot(ft,'k',linewidth=3,alpha=0.2)
plt.plot(z_st.T,'--')

plt.figure()
plt.subplot(211); plt.imshow(rt,aspect='auto')
plt.subplot(212); plt.imshow(rt_d,aspect='auto')

# %% testing
fpt = np.zeros((2,simtime_len*1))
spt = np.zeros((2,simtime_len*1))
ti = 0
x = x0
r = np.tanh(x)
z = z0
for t in range(len(simtime)*1-1):
    ti = ti+1 
    
    if t>1500:# and t<550:
        x = (1.0-dt)*x + M @ (r*dt) + 1.*wf_d#vv[:,1][:,None]#*wf_d#[:,1][:,None]#1*wf_d @ z_st[:,800][:,None] #*np.random.rand()
    else:
        x = (1.0-dt)*x + M @ (r*dt) #+ 0*wf_d[:,0][:,None]#*np.random.rand()
#    if t>1200 and t<1300:
#        x = (1.0-dt)*x + M @ (r*dt) + 1*wf_d[:,1][:,None]#*np.random.rand()
    r = np.tanh(x)
    z = wo.T @ r

    fpt[:,ti] = z.squeeze()
    e = fpt[:,ti] - ft[ti]
       

#fpt = np.squeeze(fpt)
plt.figure()
plt.plot(ft.T,'--',label='target')
plt.plot(fpt.T,label='readout')
#plt.legend(fontsize=20)
print(((fpt+0-ft)**2).sum()/(ft**2).sum())

print('Corr:', np.corrcoef(fpt.sum(0), ft)[0,1])
plt.figure()
hh = plt.xcorr(fpt.sum(0), ft,maxlags=1000)
print('lagged-corr: ', np.max(hh[1]))

# %%
plt.figure()
plt.plot(ft.sum(0),'--',label='target')
plt.plot(zpt.sum(0)+0,label='readout')
plt.legend(fontsize=20)

error_avg = sum(abs(zpt.sum(0)+0-ft.sum(0)))/simtime_len
print(['Testing MAE: ', str(error_avg.sum())]) 

plt.figure()
plt.plot(Pt_inf[:,3:].T)
plt.legend(['latent_1','latent_2'], fontsize=20)
plt.ylabel('P(state)', fontsize=40)

# %%
plt.figure()
#plt.plot(ft[0,:],'--',color='b',label='target')
#plt.plot(ft[1,:],'--',color='orange')
plt.plot(zpt[0,:],label='latent_1')#,color='blue',label='readout')
plt.plot(zpt[1,:],label='latent_2')#,color='orange')
plt.legend(fontsize=30)
