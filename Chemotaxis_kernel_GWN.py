#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:21:48 2019

@author: kschen
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import scipy.optimize   #for log-likelihood
from scipy.special import iv  #for Bessel function
from scipy.stats import vonmises  #for von Mises distribution

#incoporate the temporal kernel for chemo-sensing
def temporal_kernel(alpha,tau):
    """
    biphasic temporal kernel
    tau is the temporal domain and alpha controls the form
    """
    D_tau = alpha*np.exp(-alpha*tau)*((alpha*tau)**5/math.factorial(5) - (alpha*tau)**7/math.factorial(7))
    return D_tau
tt = np.linspace(0,10,10/0.6)
plt.plot(tt,temporal_kernel(2.,tt),'-o')

#check with auto-correlation in time series
def autocorr(x):
    """
    Autocorrelation function for time-series
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

#Hybrid Gaussian process for angle time series
def d_theta(K_dcp, dc_perp, K, K_dc, dC):
    '''
    Return change in theta angle for each step
    Input with K_dcp for kernel, dc for orthogonal concentration difference, and K covariance in weathervaning
    K_dc as the weighting/kernel on concentration in the signoidal function for tumbling rate 
    '''
    wv = np.dot(K_dcp,dc_perp) + K*np.random.randn()  #weathervaning strategy
    #P_event = 0.023/(0.4 + np.exp(40*dC/dt)) + 0.003  #sigmoidal function with parameters w
    P_event = 5*0.023/(1 + np.exp(np.dot(K_dc,dC/dt)))  #less parameter version
    if np.random.rand() < P_event:
        beta = 1
    else:
        beta = 0
    rt = beta*(np.random.rand()*360-180) #(2*np.pi)  #run-and-tumble strategy
    #rt = beta*(np.random.randn()*K + 100)  #alternative Gaussian mixture
    dth = wv + rt
    if dth > 180:
        dth = dth-360  #bounded by angle measurements
    if dth < -180:
        dth = dth+360
    return dth

#concentration gradient in space
def gradient(C0,x,y):
    """
    Gaussian sptatial profile through diffision equation
    """
    concentration = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50))  #depends on diffusion conditions along x-axis
    return concentration

#measure for concentration difference for weathervane
def dc_measure(dxy,xx,yy):
    """
    perpendicular concentration measure
    """
    perp_dir = np.array([-dxy[1], dxy[0]])  #perpdendicular direction
    perp_dir = perp_dir/np.linalg.norm(perp_dir)*1 #unit norm vector
    perp_dC = gradient(C0, xx+perp_dir[0], yy+perp_dir[1]) - gradient(C0, xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC


# %%
#gradient environment
dis2targ = 50
C0 = 0.2  #initial concentration
D = 0.000015  #diffusion coefficient (for a reasonable simulation environment)
duT = 60*60*3  #equilibrium time
d = 0.18  #difussion coefficient of butanone...

#chemotaxis strategy parameter
K_win = np.linspace(0,6,6/0.6)
scaf = 100  #scale factor
tempk = temporal_kernel(4.,K_win)/np.linalg.norm(temporal_kernel(4.,K_win))
K_dc = 100 *(tempk)+.0  #random-turning kernel (biphasic form, difference of two gammas)
#K_dc = scaf * np.flip(temporal_kernel(0.7,K_win))
#K_dc = K_dc - np.mean(K_dc)  #zero-mean kernel for stationary solution
#K_dc[np.where(K_dc>0)[0]] = 0  #rectification of the kernel
#K_dc = -np.exp(-K_win/0.5) 
wv_win = 0.5
K_dcp = scaf *np.exp(-K_win/wv_win)  #weathervaning kernel (exponential form)
K = 20  #covariance of weathervane
w = 0  #logistic parameter (default for now)
T = 3000  #whole duration of steps
dt = 0.6  #seconds
v_m = 0.12  #mm/s
v_s = 0.01  #std of speed
time = np.arange(0,T*dt,dt)
xs = np.zeros(time.shape)
ys = np.zeros(time.shape)  #2D location
prehist = max(len(K_dc),len(K_dcp))  #pre-histroy length
xs[:prehist] = np.random.randn(prehist)
ys[:prehist] = np.random.randn(prehist)
ths = np.zeros(time.shape)  #agle with 1,0
ths[:prehist] = np.random.randn(prehist)
dxy = np.random.randn(2)
dcs = np.zeros((time.shape[0],prehist))
dcps = np.zeros((time.shape[0],prehist))
dths = np.zeros(time.shape)

## with turning (Brownian-like tragectories)
for t in range(prehist,len(time)):
    
    #concentration = gradient(C0,xs[t-1],ys[t-1])
    #dC = gradient(C0, xs[t-1],ys[t-1]) - gradient(C0, xs[t-2],ys[t-2])
    #dC = np.array([gradient(C0, xs[t-past],ys[t-past])-gradient(C0, xs[t],ys[t]) for past in range(0,len(K_dc))])
    dC = np.array([gradient(C0, xs[t-past],ys[t-past]) for past in range(1,len(K_dc)+1)])
    dC = dC + np.random.randn(len(dC))*0.
    #dC = np.flip(dC)
    #dC = np.diff(dC)  #change in concentration!! (not sure if this is reasonable)
    #dc_perp = dc_measure(dxy,xs[t-1],ys[t-1])  
    dc_perp = np.array([dc_measure(dxy, xs[t-past],ys[t-past]) for past in range(1,len(K_dcp)+1)])    
    dth = d_theta(K_dcp, -dc_perp, K, K_dc, dC)
    ths[t] = ths[t-1] + dth*dt
    
    #data collection
    dcs[t,:] = dC  #concentration
    dcps[t,:] = dc_perp  #perpendicular concentration difference
    dths[t] = dth  #theta angle change
        
    e1 = np.array([1,0])
    vec = np.array([xs[t-1],ys[t-1]])
    theta = math.acos(np.clip(np.dot(vec,e1)/np.linalg.norm(vec)/np.linalg.norm(e1), -1, 1)) #current orienation relative to (1,0)

    vv = v_m + v_s*np.random.randn()
    dd = np.array([vv*np.sin(ths[t]*np.pi/180), vv*np.cos(ths[t]*np.pi/180)])  #displacement
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,s), (-s, c)))  #rotation matrix, changing coordinates
    dxy = np.dot(R,dd)
    
    xs[t] = xs[t-1] + dxy[0]*dt
    ys[t] = ys[t-1] + dxy[1]*dt

#plt.plot(ths)
plt.figure()
plt.plot(xs,ys)
plt.figure()
x = np.arange(np.min(xs),np.max(xs),1)
xx_grad = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50)) #same background environment
plt.imshow(np.expand_dims(xx_grad,axis=1).T,extent=[np.min(xs),np.max(xs),np.min(ys),np.max(ys)])
#plt.hold(True)
plt.plot(xs,ys,'white')


# %%
#####
#Inference for chemotactic strategy
#####

#von Mises distribution test (without full-model fitting)
d2r = np.pi/180
#vm_par = vonmises.fit((data_th-np.dot(data_dcp,K_dcp))*d2r, scale=1)
#plt.hist(data_th*d2r,bins=100,normed=True);

def RaisedCosine_basis(nkbins,nBases):
    """
    Raised cosine basis function to tile the time course of the response kernel
    nkbins of time points in the kernel and nBases for the number of basis functions
    """
    #nBases = 3
    #nkbins = 10 #binfun(duration); # number of bins for the basis functions
    ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.5),(nBases,1))  #take log for nonlinear time
    #ttb = np.tile(np.arange(0,nkbins),(nBases,1))
    dbcenter = nkbins / (nBases+3) # spacing between bumps
    width = 4*dbcenter # width of each bump
    bcenters = 2.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
    def bfun(x,period):
        return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)
    temp = ttb - np.tile(bcenters,(nkbins,1)).T
    BBstm = [bfun(xx,width) for xx in temp] 
    #plt.plot(np.array(BBstm).T)
    return np.array(BBstm)

#negative log-likelihood
def nLL(THETA, dth,dcp,dc):
    """
    negative log-likelihood objective function for fitting
    THETA includes parameter to be inferred and dth, dcp, dc are from recorded data
    """
    k_, A_, B_, Amp, tau = THETA[0], THETA[1], THETA[2:7], THETA[7],THETA[8] #, THETA[8]#, THETA[9] #Kappa,A,Kdc,Kdcp,dc_amp,dcp_amp
    B_ = np.dot(B_,RaisedCosine_basis(len(K_win),5))  #turning kernel (Kdc)
    #B_ = 100* B_/np.linalg.norm(B_)
    #P = sigmoid(A_, B_, C_, D_, dcp)
    P = sigmoid2(A_,B_,dc)
    rv = vonmises(k_)
    C_ = -Amp *np.exp(-K_win/tau)  #W-V kernel (Kdcp)
    VM = rv.pdf((dth-np.dot(dcp,C_))*d2r)
    marginalP = np.multiply((1-P), VM) + (1/(2*np.pi))*P
    nll = -np.sum(np.log(marginalP+1e-9))#, axis=1)
    return np.sum(nll)

def nLL2(THETA, VM, dth,dcp,dc):
    A_, B_ = THETA  #inferred paramter
    P = sigmoid2(A_, B_, dcp)
    marginalP = np.multiply((1-P), VM) + (1/(2*np.pi))*P
    nll = -np.sum(np.log(marginalP+1e-7))
    return np.sum(nll)

#sigmoidal function
def sigmoid(a,b,c,d,x):
    #a,b,c,d = p
    y = a / (b + np.exp(c*x)) + d
    ###Simulated function
    #P_event = 0.115/(1 + np.exp(40*dC/dt))
    return y

def sigmoid2(a,b,x):
    #a,b,c,d = p
    y = a / (1 + np.exp(np.dot(b,x.T/dt)))
    ###Simulated function
    #P_event = 0.023/(0.4 + np.exp(40*dC/dt)) + 0.003
    return y

###Take derivative for optimization
def der(THETA):
    a_, k_, A_, B_, C_, D_ = THETA
    der = np.zeros(len(THETA))
    der[0] = 0
    der[1] = 0 #...
    return der


# %%
#stimulate with white noise
def generate_noise(NN):
    """
    Generate dth response with white noise mapping rather than concentration environment
    """
    all_dc_p = []
    all_dc = []
    all_th = []
    for ii in range(NN):  #repetition
        prehist = max(len(K_dc),len(K_dcp))  #kernel length
        dcs = np.zeros((time.shape[0],prehist))
        dcps = np.zeros((time.shape[0],prehist))
        dths = np.zeros(time.shape)
        for t in range(prehist,len(time)):  #time series
            dC = np.random.randn(len(K_dc))*.1  #white noise for mapping
            dc_perp = np.random.randn(len(K_dcp))*.1  #white noise for mapping   
            dth = d_theta(K_dcp, -dc_perp, K, K_dc, dC) 
            #data collection
            dcs[t] = dC  #concentration
            dcps[t] = dc_perp  #perpendicular concentration difference
            dths[t] = dth  #theta angle change
    
        all_dc_p.append(dcps)  #recording dC_perpendicular
        all_dc.append(dcs)  #recording dC
        all_th.append(dths)  #recording head angle
    
    ###ALL DATA HERE~~
    data_th = np.array(all_th).reshape(-1)
    data_dcp = np.vstack(all_dc_p)
    data_dc = np.vstack(all_dc)    
    
    return data_th, data_dcp, data_dc

# %%
###generating data
#dth_n, dcp_n, dc_n = generate_noise(30)
data_th,data_dcp,data_dc = generate_noise(30)

# %%
#optimize all with less parameters
theta_guess = np.array([100,0.1])  #Kappa, A
theta_guess = np.concatenate((theta_guess,np.random.randn(5)))  #random weight for basis of Kdc kernel
theta_guess = np.concatenate((theta_guess,np.array([0.5,10])))  #tau,dcp_amp
#theta_guess = np.concatenaate((theta_guess, theta_fit[3:]))  #use a "good" inital condition from the last fit
###Ground Truth: 25,5,0.023,0.4,40,0.003
###k_, A_, a_N, a_exp, B_N, B_exp = 25, 5, 30, 4, 30, 0.5
res = scipy.optimize.minimize(nLL,theta_guess,args=(data_th,data_dcp,data_dc))#,method='Nelder-Mead')

theta_fit = res.x

# %%
### check kernel forms!!
fit_par = theta_fit[2:7]
recKdc = np.dot(fit_par,RaisedCosine_basis(len(K_dc),len(fit_par)))  #reconstruct Kdc kernel
#recKdc = recKdc/np.linalg.norm(recKdc)
plt.plot(recKdc,'b',label='K_c_fit',linewidth=3)
plt.plot(K_dc,'b--',label='K_c',linewidth=3)  #compare form with normalized real kernel
plt.figure()
recKdcp = theta_fit[7]*np.exp(-K_win/theta_fit[8])  #reconstruct Kdcp kernel
plt.plot(recKdcp,'r',label='K_cp_fit',linewidth=3)
#plt.hold(True)
plt.plot(K_dcp,'r--',label='K_cp',linewidth=3)
plt.legend()

# %%
###check sigmoid curve
plt.figure()
xp = np.linspace(-0.5, 0.5, 1000)
rescl =np.linalg.norm(K_dc)/np.linalg.norm(recKdc)  #use this before learning the scale factor (it will be exactly the scale factor above)
conv_dc1 = np.dot(recKdc*rescl,dc_n.T)
pxp = sigmoid2(theta_fit[1],recKdc*rescl,dc_n)
plt.plot(conv_dc1,pxp,'o',linewidth=3,label='inferred',color='r',alpha=0.1)
conv_dc = np.dot(K_dc,dc_n.T)
plt.plot(conv_dc, sigmoid2(5*0.023, K_dc,dc_n),'o',linewidth=5,label='true',alpha=0.1)
plt.xlabel('x')
plt.ylabel('y',rotation='horizontal') 
plt.grid(True)
plt.legend()

### check on von Mises density
plt.figure()
aa,bb = np.histogram((data_th-np.dot(data_dcp,recKdcp))*d2r,bins=200)
plt.bar(bb[:-1],aa/len(data_th),align='edge',width=0.03,label='true')
rv = vonmises(res.x[0])
#plt.scatter((data_th-alpha*data_dcp)*d2r,rv.pdf((data_th-alpha*data_dcp)*d2r),s=1,marker='.')
plt.bar(bb[:-1],rv.pdf(bb[:-1])*np.mean(np.diff(bb)),alpha=0.5,align='center',width=0.03,color='r',label='inferred')
plt.axis([-.5,.5,0,0.5])
plt.legend()
plt.xlabel('heading')
plt.ylabel('pdf')
#normalization by bin size???
#checking pdf density
print('sum of histogram:',np.sum(aa/len(data_th)))
print('integrate von Mises:',np.sum(rv.pdf(bb[:-1])*np.mean(np.diff(bb))))

# %%
###
###Scanning over data length and observe convergence of MSE
Ns = np.array([10,30,50,70,90])
Ns = np.array([5,5,5,5,5])
all_theta_fit = []
MSEs = []
for nn in Ns:
    print(nn)
    dth_n, dcp_n, dc_n = generate_noise(nn)
    res = scipy.optimize.minimize(nLL,theta_guess,args=(data_th,data_dcp,data_dc))#,method='Nelder-Mead')
    theta_fit = res.x
    fit_par = theta_fit[2:7]
    recKdc = np.dot(fit_par,RaisedCosine_basis(len(K_dc),len(fit_par)))  #reconstruct Kdc kernel
    recKdcp = theta_fit[7]*np.exp(-K_win/theta_fit[8])  #reconstruct Kdcp kernel
    MSE_dc = np.sum((K_dc-recKdc)**2)
    MSE_dcp = np.sum((K_dcp-recKdcp)**2)
    
    all_theta_fit.append(res.x)  #all theta_fit
    MSEs.append([MSE_dc,MSE_dcp])  #all MSE measured for two kernels

