# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 02:05:30 2022

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Try fFORCE learning with state-dependent network feedback,
with local learning rules for state,
and trying to learn feedback noise for transitions!
... need to figure out noise strength and target denoising~~
... OR figure out orthogonal dynamics!
... OR try contrastive Hebbian learning or others~

"""

import numpy as np
import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from autograd import grad, jacobian
from scipy.optimize import minimize

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

#%matplotlib qt5

# %% switching time series
def Lorenz(x,dt,params):
    return dx

def Complex_p(x,dt,params):
    return dx

def Switch(lt, M):
    return st

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

def network_cluster(Ns, ps, gs):
    N_in, N_ch = Ns
    p_in, p_ch = ps
    g_in, g_ch = gs
    M_in = network(N_in, p_in, g_in)
    M_ch = network(N_ch, p_ch, g_ch)
    
    M_ic = np.random.randn(N_in, N_ch)*.1  #test for feed-forward structure
    sparse = np.random.rand(N_in, N_ch)
    mask = sparse*0
    p = 0.2
    mask[sparse>p] = 0
    mask[sparse<=p] = 1
    M_ic = M_ic*mask
    J = np.block([[M_in, M_ic*0],[M_ic.T, M_ch]])
    return J

def stim_ssm():
    amp = 1
    freq = 1/20;
    f_s = 1/150
    ft_ = amp*np.sin(np.pi*f_s*simtime)
    high_f = amp*np.sin(np.pi*freq*simtime*2.)
    low_f = amp*np.sin(np.pi*freq*simtime*.5)
    offset = .5
    pos_h = np.where(ft_>offset)[0]
    pos_l = np.where(ft_<=offset)[0]
    ft = np.zeros((2,simtime_len))
    ft[0,pos_h] = high_f[pos_h]*1.5
    ft[1,pos_l] = low_f[pos_l]*1
    zt = np.zeros((2,simtime_len))
    zt[0,pos_h] = 1
    zt[1,pos_l] = 1
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

def log_trans(ws,st,r):
    w1 = ws[:int(len(ws)/2)]
    w2 = ws[int(len(ws)/2):]
    Pt = np.array([np.exp(w1 @ r),np.exp(w2 @ r)])[:,0]
    Pt /= sum(Pt)
    logp = np.dot(st,np.log(Pt))
    return -logp

# %% parameters
### network  (Nin+Nch x Nin+Nch)
N_in, N_ch = 100, 200
Ns = N_in, N_ch
ps = 0.2, 0.5
gs = .9, 1.5
alpha = 1.  #learning initial constant
dt = 0.1
nsecs = 1000
learn_every = 2  #effective learning rate
simtime = np.arange(0,nsecs,dt)
simtime_len = len(simtime)

M = network_cluster(Ns, ps, gs)
#M[:N_in,:N_in] = wf_s[:N_in,:] @ wf_s[:N_in,:].T + 2*wf_s[:N_in,0][:,None] @ wf_s[:N_in,1][:,None].T
M_d = M*1#

### target readout  (Nin+Nch x signal-dimension)
N = N_in + N_ch
nRec2Out = (N,1)
uu,ss,vv = np.linalg.svd(np.random.randn(N_in,N_in))
wo = np.zeros(nRec2Out)
dw = np.zeros(nRec2Out)
wf = np.concatenate((2.0*(np.random.rand(N_in,1)-0.5)*.1 , np.zeros((N_ch,1))),0)
#wf = np.concatenate((uu[:,0][:,None] , np.zeros((N_ch,1))),0)

### symbol readout  (Nin+Nch x state)
wo_s = 2.0*(np.random.rand(N_in,2)-0.5)*.1
dw_s = dw*1
wf_s = np.concatenate((2.0*(np.random.rand(N_in,2)-0.5)*.1 , np.zeros((N_ch,2))),0)
#wf_s = np.concatenate((2.0*(np.random.rand(N_in,2)-0.5)*.1 , 2.0*(np.random.rand(N_ch,2)-0.5)*.1),0)
#wf_s = np.concatenate((uu[:,1:3] , np.zeros((N_ch,2))),0)

wf_h = np.concatenate((uu[:,4][:,None] , np.zeros((N_ch,1))),0)

### driven readout  (Nch x states)... use this to learn to drive state transitions!
wo_d = 2.0*(np.random.rand(N_ch,1)-0.5)*.1
dw_d = wo_d*0
wf_d = wf_s.sum(1)[:,None]#np.concatenate((wf_s.sum(1)[:,None],wf_s.sum(1)[:,None]),1)
#wf_d = np.concatenate((2.0*(np.random.rand(N_in,2)-0.5)*.1 , np.zeros((N_ch,2))),0)
#wf_d = np.concatenate((uu[:,1:3][:,None] , np.zeros((N_ch,2))),0)

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
z_s = np.zeros((2,1))

mask = np.ones((N,N))*0
mask[N_in:,N_in:] = 1
wo_s_ = wo_s*1

connectom = np.ones((N_ch,N_ch))
pos = np.where(M[N_in:,N_in:]==0)
connectom[pos[0],pos[1]] = 0

noise = .00
mask_n = np.zeros((N,1))
mask_n[:N_in,0] = 1

M_in = M[:N_in, :N_in]
#s = 1.2
#np.fill_diagonal(M, s)
#M[:N_in, :N_in] = M_in
#M[:N_in, :N_in] = 1/N*(np.cov(wf_s[:N_in,:]) - 1)
#gam1, gam2, gam3 = 2,20,50
gam1, gam2, gam3 = .5,5,50
#.2, 5, 10.
syn = 2
tau_m = 1000

es = np.zeros(simtime_len)
# %% FORCE learning
z_ft = np.zeros((1,simtime_len))
z_st = np.zeros((2,simtime_len))
plt.figure()
ti = 0
P = (1.0/alpha)*np.eye(N)
P_d = P*1
P_ = (1.0/alpha)*np.eye(N_ch)
P2 = (1.0/1)*np.eye(N_in)
Pw = np.concatenate((P2[:,:,None],P2[:,:,None]),2)
for t in range(len(simtime)-1):
    ti = ti+1
    noiset = noise*np.random.randn(N,2)*mask_n
    stt = np.array([[0.],[1.]]) #st[:,ti][:, None]            #use the training state
    ### target network
    r_ = r  #last step
    z_s = wo_s.T @ r[:N_in,:]
    fmax = np.zeros((2,1))
    fmax[np.argmax(z_s)] = 1
    z_d = wo_d.T @ r[N_in:,:]
    hint = (st[:,t] - st[:,t-1])[:,None]*1
    hintt = (st[0,t] - st[0,t-1])
    x = (1.0-dt)*x + M @ (r*dt) + 1*wf_s @ stt + 0*wf_d*hintt + (wf_s*noiset).sum(1)[:,None]#@ z_s #
    r = np.tanh(x)
    rt[:,t] = r[:,0]
    z = wo.T @ r  # linear decoding
    inpt = np.array([ft[ti]])[:,None]  #either use this or z for driving
    
    ### driven network
    x_d = (1.0-dt)*x_d + M_d @ (r_d*dt) + wf @ inpt + 1*wf_s @ stt \
            + 0*wf_d*hintt + wf_h*hintt + (wf_s*noiset).sum(1)[:,None] # z_s #
    r_d = np.tanh(x_d)
    rt_d[:,t] = r_d[:,0]
    
    if np.mod(ti, learn_every) == 0:
        ### RLS calculation
        P, k, c = RLS(P, r)
        P_, k_, c_ = RLS(P_, r[N_in:,:])
        P2, k2, c2 = RLS(P2, r[:N_in,:])
        
        ### error terms
        e = z - ft[ti]
        e_s = z_s - ( 1.*st[:,ti][:,None]-0.) #- z
        e_d = z_d - hintt*2
        e_J = M[N_in:,N_in:] @ r[N_in:, :] - M_d[N_in:,N_in:] @ r_d[N_in:, :] - wf[N_in:, :] @ inpt #wo.T @ r #- wf_s @ fmax #z_s#- z_d #*st[0,ti]#w_h*hint*0
#        e_J = M @ r - M_d @ r_d - wf @ inpt #
        
        ### update weights
        dw = -k*c*e.T
        wo = wo + dw  # target readout
        dJ = -e_J @ (P_ @ r[N_in:,:]).T
#        dJ = -e_J @ (P_ @ r).T
#        dJ = dJ*mask
        M[N_in:,N_in:] += (dJ)# #*connectom    # target network
#        M[:N_in, N_in:] = 0
#        M[N_in:, N_in:] *=  connectom
           
        ### state updates
        Pt = np.array([np.exp(wo_s[:,0] @ r[:N_in,:])+0., np.exp(wo_s[:,1] @ r[:N_in,:])])
#        Pt = np.array([np.exp(wo_d[:,0] @ r[N_in:,:])+0., np.exp(wo_d[:,1] @ r[N_in:,:])])
#        Pt = np.array([np.exp(x.T @ (np.outer(wf_s[:,0] , wo_s[:,0])) @ r) + 0., \
#                       np.exp(x.T @ (np.outer(wf_s[:,1] , wo_s[:,1])) @ r)] )[:,:,0]  # energy form!
        Pt /= sum(Pt)
#        dw_s = -k*c*(1*e_s).T*1 -P @ (r) @ ((1-Pt)).T*st[:,ti]
#        dw_s = -(P2 @ (r[:N_in,:]) @ ((1-Pt)).T) * st[:,ti]
        dw_s = -k2*c2*((1*e_s).T)  #-k*c*((1*e_s).T)  #
        wo_s = wo_s + dw_s  # state readout
#        M[:N_in, :N_in] += wf_s[:N_in,:] @ wo_s.T
        
        ### noise transition update
#        dw_d = (P_ @ (r[N_in:,:]) @ ((1-Pt)).T)
#        dw_d = -k_*c_*((1*e_d).T)  #-k*c*((1*e_s).T)  #
#        wo_d = wo_d + dw_d
        
        ### transition log
#        for ww in range(2):
#            lamb = 1 - Pt[ww]*(1-Pt[ww])#Pt[ww]*(1 - Pt[ww])
#            k = Pw[:,:,ww] @ r[:N_in,:]
#            rPr = r[:N_in,:].T @ k
#            c = 1.0/(1/lamb + 1*rPr)
#            Pw[:,:,ww] = 1/1*(Pw[:,:,ww] - 1*k @ (k.T * c)) 
##            Pw[:,:,ww] = P
##        dw_s =  np.einsum("ijk,jk->ik", Pw, -r*(1*e_s).T*0 + 1*r*((1-Pt)).T)#1*r*((1-Pt)*(st[:,ti][:,None])).T)
#        dw_s =  np.einsum("ijk,jk->ik", Pw, -r[:N_in,:]*(1*e_s).T*1 + 1*r[:N_in,:]*((1-Pt)*(st[:,ti][:,None])).T)
##        dw_s = np.einsum("ijk,jk->ik", Pw, r*(1*e_s).T*0 + (x.T @ wf_s[:,ww][:,None] * r)*((1-Pt)*(st[:,ti][:,None])).T)
##        wo_s_ = wo_s
#        wo_s = wo_s + dw_s  # state readout
        
        temp = M[:N_in, :N_in] + 0*dt*( -(M[:N_in, :N_in] - M_in)/tau_m  + 1**2*(\
                          - 1*gam1/N* (r[:N_in,:] @ r[:N_in,:].T).T  \
                          + 0*gam2/N*wf_s[:N_in,:] @  np.diag(stt[:,0]) @ wf_s[:N_in,:].T  \
                          + 1*gam2/N*r[:N_in,:] @ (stt[:,0] @ wf_s[:N_in,:].T)[:,None].T
                          + 1*gam3/N *np.abs(hintt)*(np.outer(wf_s[:N_in,0],wf_s[:N_in,1]).T) \
                          + 1*gam3/N *np.abs(hintt)*(np.outer(wf_s[:N_in,1],wf_s[:N_in,0]).T) \
                          + 0*gam3/N *np.abs(hintt)*(r_[:N_in,:] @ r[:N_in,:].T).T ) )
        
#        temp = M[:N_in, :N_in] + dt*( -(M[:N_in, :N_in] - M_in)/tau_m  + 1**2*(\
#                          - 1*gam1/N* r[:N_in,:] @ (stt[:,0] @ wf_s[:N_in,:].T)[:,None].T  \
#                          + 1*gam2/N* wf_s[:N_in,:] @  np.diag(stt[:,0]) @ wf_s[:N_in,:].T \
#                          + 1*gam3/N *np.abs(hintt)*(np.outer(wf_s[:N_in,1],wf_s[:N_in,0]).T)) )
        ##### IDEA: incoperate error signal1!
#        temp = syn*np.tanh(temp/syn)
        M[:N_in, :N_in] = temp
        

    # Store the output of the system.
    z_ft[:,ti] = np.squeeze(z)
    z_st[:,ti] = fmax.squeeze() #z_s
    es[ti] = e.squeeze()**2

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
#M[:N_in, :N_in] += 10/N*(np.outer(wf_s[:N_in,1],wf_s[:N_in,0]))
rpt = np.zeros((N,simtime_len))
mult = 1
noise = .8   ### try correlated noise~~
fpt = np.zeros((1,simtime_len*mult))
spt = np.zeros((2,simtime_len*mult))
ti = 0
x = x0
r = np.tanh(x)
z = z0
match = [1,0]
theta = np.concatenate((0.*np.ones((N_in,1)) , np.zeros((N_ch,1))),0)
for t in range(len(simtime)*mult-1):
    ti = ti+1 
    
    z_s = wo_s.T @ r[:N_in,:]
    fmax = np.zeros((2,1))
    fmax[np.argmax(z_s)] = 1
    noiset = noise*np.random.randn(N,1)*mask_n  #()
    z_d = wo_d.T @ r[N_in:,:]
    ### change into Markovian??
    
    ### perturbation dynamics
    x = (1.0-dt)*x + M @ (r*dt) + 0*wf_s @ fmax + 1*wf_s[:,[int(st[0,t])]] \
        + 0*noiset*(wf_s @ np.array([[1,1]]).T) + 0*noiset*wf_d  # + 0*wf_d @ z_d

    r = np.tanh(x + theta)
    z = wo.T @ r

    fpt[:,ti] = z.squeeze()
    spt[:,ti] = z_s.squeeze() #z_s
    rpt[:,ti] = r.squeeze()
       

#fpt = np.squeeze(fpt)
plt.figure()
plt.plot(ft,'k',linewidth=3,alpha=0.2)
plt.plot(fpt.T,label='readout')
#plt.plot(np.convolve(fpt[0,:],np.ones(50)/50,'same'),label='readout')
plt.plot(spt.T,'--',label='state')
#plt.legend(fontsize=20)
print(((fpt+0-ft)**2).sum()/(ft**2).sum())

print('Corr:', np.corrcoef(fpt.sum(0), ft)[0,1])
plt.figure()
hh = plt.xcorr(fpt.sum(0), ft,maxlags=1000)
print('lagged-corr: ', np.max(hh[1]))
