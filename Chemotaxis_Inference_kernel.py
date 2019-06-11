#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 15:50:11 2019

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
    concentration = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50))  #depends on diffusion conditions
    return concentration

#measure for concentration difference for weathervane
def dc_measure(dxy,xx,yy):
    """
    perpendicular concentration measure
    """
    perp_dir = np.array([-dxy[1], dxy[0]])  #perpdendicular direction
    perp_dir = perp_dir/np.linalg.norm(perp_dir)
    perp_dC = gradient(C0, xx+perp_dir[0], yy+perp_dir[1]) - gradient(C0, xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

#gradient environment
dis2targ = 50
C0 = 0.2  #initial concentration
D = 0.000015  #diffusion coefficient (for a reasonable simulation environment)
duT = 60*60*3  #equilibrium time
d = 0.18  #difussion coefficient of butanone...

#chemotaxis strategy parameter
K_win = np.linspace(0,6,6/0.6)
scaf = 50  #scale factor
K_dc = scaf *(temporal_kernel(4.,K_win))+.0  #random-turning kernel (biphasic form, difference of two gammas)
#K_dc = K_dc - np.mean(K_dc)  #zero-mean kernel for stationary solution
#K_dc[np.where(K_dc>0)[0]] = 0  #rectification of the kernel
#K_dc = -np.exp(-K_win/0.5) 
wv_win = 0.5
K_dcp = scaf *np.exp(-K_win/wv_win)  #weathervaning kernel (exponential form)
K = 5  #covariance of weathervane
w = 0  #logistic parameter (default for now)
T = 5000  #whole duration of steps
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
    dC = np.array([gradient(C0, xs[t-past],ys[t-past]) for past in range(0,len(K_dc))])
    #dC = np.flip(dC)
    #dC = np.diff(dC)  #change in concentration!!
    #dc_perp = dc_measure(dxy,xs[t-1],ys[t-1])  
    dc_perp = np.array([dc_measure(dxy, xs[t-past],ys[t-past]) for past in range(0,len(K_dcp))])    
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

#####
#Generate trajeectories
#####
def generate_traj(NN):
    all_dc_p = []
    all_dc = []
    all_th = []
    for ii in range(NN):
        xs = np.zeros(time.shape)
        ys = np.zeros(time.shape)  #2D location
        xs[0] = np.random.randn()*0.1
        ys[0] = np.random.randn()*0.1
        prehist = max(len(K_dc),len(K_dcp))  #pre-histroy length
        xs[:prehist] = np.random.randn(prehist)
        ys[:prehist] = np.random.randn(prehist)
        ths = np.zeros(time.shape)  #agle with 1,0
        ths[:prehist] = np.random.randn(prehist)
        dxy = np.random.randn(2)
        dcs = np.zeros((time.shape[0],prehist))
        dcps = np.zeros((time.shape[0],prehist))
        dths = np.zeros(time.shape)
        for t in range(prehist,len(time)):
            
            #concentration = gradient(C0,xs[t-1],ys[t-1])
            #dC = gradient(C0, xs[t-1],ys[t-1]) - gradient(C0, xs[t-2],ys[t-2])
            #dC = np.array([gradient(C0, xs[t-past],ys[t-past])-gradient(C0, xs[t],ys[t]) for past in range(0,len(K_dc))])
            dC = np.array([gradient(C0, xs[t-past],ys[t-past]) for past in range(0,len(K_dc))])
            #dC = np.flip(dC)
            #dC = np.diff(dC)  #change in concentration!!
            #dc_perp = dc_measure(dxy,xs[t-1],ys[t-1])  
            dc_perp = np.array([dc_measure(dxy, xs[t-past],ys[t-past]) for past in range(0,len(K_dcp))])    
            dth = d_theta(K_dcp, -dc_perp, K, K_dc, dC)
            ths[t] = ths[t-1] + dth*dt
            
            #data collection
            dcs[t] = dC  #concentration
            dcps[t] = dc_perp  #perpendicular concentration difference
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
    
        all_dc_p.append(dcps)  #recording dC_perpendicular
        all_dc.append(dcs)  #recording dC
        all_th.append(dths)  #recording head angle
            
        ## plt.plot(xs,ys)
    
    ###ALL DATA HERE~~
    data_th = np.array(all_th).reshape(-1)
    #data_dcp = np.array(all_dcp).reshape(-1)
    data_dcp = np.vstack(all_dc_p)
    #data_dc = np.array(all_dc).reshape(-1)
    data_dc = np.vstack(all_dc)    
    
    return data_th, data_dcp, data_dc


#####
#Inference for chemotactic strategy
#####

#von Mises distribution test (without full-model fitting)
d2r = np.pi/180
#vm_par = vonmises.fit((data_th-np.dot(data_dcp,K_dcp))*d2r, scale=1)
#plt.hist(data_th*d2r,bins=100,normed=True);
#plt.hold(True)
#xx = np.linspace(np.min(data_th*d2r),np.max(data_th*d2r),100)
#rv = vonmises(1/theta_fit[0]**0.5)#(vm_par[0])
#plt.plot(xx, rv.pdf(xx),linewidth=3)

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
    #a_, k_, A_, B_, C_, D_ = THETA  #inferred paramter
    k_, A_, B_, C_,Bamp = THETA[0], THETA[1], THETA[2:7], THETA[7], THETA[8]#, THETA[9] #Kappa,A,Kdc,Kdcp,dc_amp,dcp_amp
    B_ = Bamp *np.dot(B_,RaisedCosine_basis(len(K_win),5))  #test with basis function
    #P = sigmoid(A_, B_, C_, D_, dcp)
    P = sigmoid2(A_,B_,dc)
    #VM = np.exp(k_*np.cos((dth-a_*dcp)*d2r)) / (2*np.pi*iv(0,k_))#von Mises distribution
    #vm_par = vonmises.fit((dth-a_*dcp)*d2r, scale=1)
    rv = vonmises(k_)#(vm_par[0])
    C_ = scaf *np.exp(-K_win/C_)
    VM = rv.pdf((dth-np.dot(dcp,C_))*d2r)
    marginalP = np.multiply((1-P), VM) + (1/(2*np.pi))*P
    nll = -np.sum(np.log(marginalP+1e-9))#, axis=1)
    #fst = np.einsum('ij,ij->i', 1-P, VM)
    #snd = np.sum(1/np.pi*P, axis=1)
    return np.sum(nll)

#rv = vonmises(K)
#VM = rv.pdf((data_th-np.dot(data_dcp,K_dcp))*d2r)
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
    #P_event = 0.023/(0.4 + np.exp(40*dC/dt)) + 0.003
    return y

def sigmoid2(a,b,x):
    #a,b,c,d = p
    y = a / (1 + np.exp(np.dot(x,b/dt)))
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

###generating data
data_th, data_dcp, data_dc = generate_traj(50)
#optimize all with less parameters
theta_guess = np.array([100,0.1])  #Kappa, A
theta_guess = np.concatenate((theta_guess,np.random.randn(5)))  #random weight for basis of Kdc kernel
# theta_guess = np.concatenate((theta_guess,np.random.randn(5)))
theta_guess = np.concatenate((theta_guess,np.array([0.5,50])))  #tau,dc_amp,dcp_amp
#theta_guess = np.concatenate((theta_guess, theta_fit[3:]))  #use a "good" inital condition from the last fit
###Ground Truth: 25,5,0.023,0.4,40,0.003
###k_, A_, a_N, a_exp, B_N, B_exp = 25, 5, 30, 4, 30, 0.5
res = scipy.optimize.minimize(nLL,theta_guess,args=(data_th,data_dcp,data_dc),method='Nelder-Mead')
                              #,bounds = ((0,None),(0,None),(None,None),(None,None)))
theta_fit = res.x
##optimize logistic
####Ground Truth: 25,5,0.023,0.4,40,0.003
#res = scipy.optimize.minimize(nLL2,theta_guess,args=(VM,data_th,data_dcp,data_dc),bounds = ((0,None),(0,None)))


### check kernel forms!!
fit_par = theta_fit[2:7]
recKdc = np.dot(fit_par,RaisedCosine_basis(len(K_dc),len(fit_par)))  #reconstruct Kdc kernel
recKdc = recKdc/np.linalg.norm(recKdc)
plt.plot(recKdc,'b',label='K_c_fit',linewidth=3)
plt.plot(K_dc/np.linalg.norm(K_dc),'b--',label='K_c',linewidth=3)  #compare form with normalized real kernel
fit_par2 = theta_fit[7]  #single exponent fit
# recKdcp = np.dot(fit_par2,RaisedCosine_basis(len(K_dcp),len(fit_par2)))  #for basis functions
recKdcp = np.exp(-K_win/theta_fit[7])
plt.plot(recKdcp/np.linalg.norm(recKdcp),'r',label='K_cp_fit',linewidth=3)
#plt.hold(True)
plt.plot(K_dcp/np.linalg.norm(K_dcp),'r--',label='K_cp',linewidth=3)
plt.legend()


### more repetition ###
allest = []
tt = 0
for sim in range(10):  #repeat simulation
    data_th, data_dcp, data_dc = generate_traj(100)
    for rep in range(3):  #repeat fit
        theta_guess = np.array([100,0.1,0.5])  #Kappa, A, kernal_parameter
        theta_guess = np.concatenate((theta_guess,np.random.randn(len(K_dc)-7)))  #the remaining parameters for weighted basis
        res = scipy.optimize.minimize(nLL,theta_guess,args=(data_th,data_dcp,data_dc),method='Nelder-Mead')
        theta_fit = res.x
        reconK = np.dot(theta_fit[3:],RaisedCosine_basis(len(K_dc),len(theta_guess)-3))
        plt.plot(reconK/np.linalg.norm(reconK))
        plt.plot(K_dc/np.linalg.norm(K_dc))
        allest.append(reconK)  #storing all reconstructed kernels
        tt = tt+1
        print(tt)
k = [i/np.linalg.norm(i) for i in allest]
plt.errorbar(range(10),np.mean(k,axis=0),yerr=np.std(k,axis=0))
### checking optimization fits
#plt.plot(theta_fit[2:2+len(K_dc)]/np.linalg.norm(theta_fit[2:2+len(K_dc)]),label='K_c',linewidth=3)
#plt.hold(True)
#plt.plot(K_dc/np.linalg.norm(K_dc),'b--',label='K_c fit',linewidth=3)
#plt.plot(theta_fit[-len(K_dcp):]/np.linalg.norm(theta_fit[-len(K_dcp):]),'r',label='K_cp',linewidth=3)
##plt.hold(True)
#plt.plot(K_dcp/np.linalg.norm(K_dcp),'r--',label='K_cp',linewidth=3)
#plt.legend()

#plt.plot(30*(temporal_kernel(theta_fit[2], K_win)),label='K_c_fit',linewidth=3)
#plt.hold(True)
#plt.plot(K_dc,'b--',label='K_c',linewidth=3)
#plt.plot(30*np.exp(-K_win/theta_fit[3]),'r',label='K_cp_fit',linewidth=3)
##plt.hold(True)
#plt.plot(K_dcp,'r--',label='K_cp',linewidth=3)
#plt.legend()


###############################
### check on von Mises density
#plt.hist((data_th-alpha*data_dcp)*d2r,bins=100,normed=True,color='r');
aa,bb = np.histogram((data_th-np.dot(data_dcp,recKdcp))*d2r,bins=200)
plt.bar(bb[:-1],aa/len(data_th),align='edge',width=0.03)

rv = vonmises(res.x[0])
#plt.scatter((data_th-alpha*data_dcp)*d2r,rv.pdf((data_th-alpha*data_dcp)*d2r),s=1,marker='.')
plt.bar(bb[:-1],rv.pdf(bb[:-1])*np.mean(np.diff(bb)),alpha=0.5,align='center',width=0.03,color='r')
plt.axis([-.5,.5,0,0.5])
#normalization by bin size???
#checking pdf density
print('sum of histogram:',np.sum(aa/len(data_th)))
print('integrate von Mises:',np.sum(rv.pdf(bb[:-1])*np.mean(np.diff(bb))))

### check on logistic fitting
xp = np.linspace(-0.5, 0.5, 1000)
pxp=sigmoid2(res.x[2],np.dot(theta_fit[2:7],RaisedCosine_basis(len(K_dc),5)),data_dc)
#pxp=sigmoid1(res.x[2],res.x[3],res.x[4],res.x[5],xp)
plt.plot(xp,pxp,'-',linewidth=3,label='fit')
#plt.hold(True)
plt.plot(xp,sigmoid2(5*0.023, 140/0.6,xp),linewidth=5,label='ground-truth',alpha=0.5)
#rescl = max(K_dc)/max(recKdc)  #use this before learning the scale factor...
rescl = theta_fit[-1]
conv_dc1 = np.dot(recKdc*rescl,data_dc.T)
pxp = sigmoid2(theta_fit[1],recKdc*rescl,data_dc)
#pxp=sigmoid1(res.x[2],res.x[3],res.x[4],res.x[5],xp)
plt.plot(conv_dc1,pxp,'o',linewidth=3,label='fit')
#plt.plot(xp,sigmoid2(5*0.023, 140/0.6,xp),linewidth=5,label='ground-truth',alpha=0.5)
conv_dc = np.dot(K_dc,data_dc.T)
plt.plot(conv_dc, sigmoid2(5*0.023, K_dc,data_dc),'o',linewidth=5,label='ground-truth',alpha=0.5)
#plt.plot(xp,sigmoid1(0.023,0.4,140/0.6,0.003,xp),linewidth=3,label='ground-truth')
plt.xlabel('x')
plt.ylabel('y',rotation='horizontal') 
plt.grid(True)
plt.legend()

### step-wise fitting
#...