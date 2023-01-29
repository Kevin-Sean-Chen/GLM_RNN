#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:01:06 2019

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
from scipy.stats import uniform
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20)

#from Chemotaxis_Inference_kernel import temporal_kernel, autocorr, gradient,\
#                                        RaisedCosine_basis, sigmoid2, generate_traj

# %%
### paramerers
#behavior
v_abs = 0.22
theta_dot = 10
tau0 = 4.2
sigma = 5#np.pi/6

#navigation
tau0 = 4.2  #stearing baseline
gamma0 = 0.5  #turning baseline
lamb0 = 0.03  #pirouette
alpha_s = -0.015
alpha_p = 7.5*10**-5
alpha_g = 0.006

#neural
taum = 2
V0 = -35
gdL = 100  #conductance of depolorization on the Left
gdR = 100
ghL = 100
ghR = 100
Vd = 64
Vh = -70

#channels
ad = 2
ah = 2
betadLR = 0.1
betahLR = 0.1
deltadLR = 0.05
gammadLR = 0.7
lamdL = 0.01
lamhL = 0.01
# alphahL = 0  #alpha is driven by sensory input
# alphahR = 0
alphah0 = 0.01
etaR = 60  #threshold

# %%
### behavior
def steer(V):
    return tau0/2 + alpha_s*V
def pirouette(V):
    return lamb0 + alpha_p*V
def turn(V):
    return gamma0 + alpha_g*V
def f_theta(theta, gammaB):
    gammaA = 1-gammaB
    ft = gammaA/(2*np.pi) + gammaB/(2*np.pi*sigma**2)**0.5*np.exp(-(theta-np.pi)/(2*sigma**2))
    return ft

#let us simplify it as LN model for now!
def d_theta(K_dcp, dc_perp, K, K_dc, dC):  #(dC,dCp,dth):
    '''
    Return change in theta angle for each step
    Input dC for tangent concentration difference,dCp for perpdendicular,and K covariance in weathervaning
    K_
    '''
    wv = np.dot(K_dcp,dc_perp) + K*np.random.randn()  #weathervaning strategy
    #P_event = 0.023/(0.4 + np.exp(40*dC/dt)) + 0.003  #sigmoidal function with parameters w
    P_event = 5*0.023/(1 + np.exp(np.dot(K_dc,dC/dt)))  #less parameter version
    if np.random.rand() < P_event:
        beta = 1
    else:
        beta = 0
    gammaB = 0.5   #can later be input-dependent in the future...
    #gammaB = max(0,np.dot(K_dc,dC)*0.1)  #tangent experience
    #gammaB = max(1,gammaB)  #tangent experience
    
    ###just another von Mises
    #ft = np.random.vonmises(np.pi,1)/np.pi*180   #can be input dependent in the future...
    ###mixture of von Mises and uniform that depends on input
    ft = VM_U(gammaB,sigma)*180/np.pi #bias_turn(gammaB)*180/np.pi
    rt = beta*ft  #run-and-tumble strategy with a biased turning angle
    dth = wv + rt
    if dth > 180:
        dth = dth-360  #bounded by angle measurements
    if dth < -180:
        dth = dth+360
    return dth

def VM_U(gammaB,sigma):
    """
    directly sample from uniform and von Mises
    """
    bern = np.random.binomial(1,gammaB)
    angle = (1-bern)*(np.random.rand()-0.5)*np.pi*2 + bern*np.random.vonmises(np.pi,sigma)  #mixture of uniform and VM
    return angle

def ppf_theta(gammaB,rand):
    """
    inverse of the uniform plus von Mises distribution
    """
    #return (1-gammaB)/(2*np.pi)*theta + gammaB*vonmises.ppf(theta,sigma,loc=0,scale=1)
    return (1-gammaB)*2*np.pi*rand + gammaB*vonmises.ppf(rand,sigma,loc=np.pi,scale=1)  #np.pi as the opposite direction
    
def bias_turn(gammaB):
    """
    inverse cumulated distribution sampling for the mixture of von Mises and uniform distribution
    """
    u = np.random.rand()
    return ppf_theta(gammaB,u)#-np.pi #add pi to be centered at pi rather than zero
#plt.hist(np.array([bias_turn(1.) for ii in range(1000)]),20);  #the correct mixture distribution!

# %%%%%%%%%%%%%%
def velocity_dependent(Ct,vmax,k,m):
    """
    Hill function-like modulation of velocity
    ...can be fit separately with max-likelihood
    """
    return vmax*(1-Ct**m/(k**m+Ct**m))

# %%
k = 1. #half-concentration of the Hill function
m = 3  #power in Hill function
vmax = 0.3  #maximum velocity
#chemotaxis strategy parameter
K_win = np.linspace(0,6,6/0.6)
scaf = 100  #scale factor
tempk = temporal_kernel(4.,K_win)/np.linalg.norm(temporal_kernel(4.,K_win))
K_dc = 100 *(tempk)+.0  #random-turning kernel (biphasic form, difference of two gammas)
wv_win = 0.5
K_dcp = scaf *np.exp(-K_win/wv_win)  #weathervaning kernel (exponential form)
K = 5  #covariance of weathervane
w = 0  #logistic parameter (default for now)
T = 1000  #whole duration of steps
dt = 0.6  #seconds
v_m = 0.12  #mm/s
v_s = 0.01  #std of speed
time = np.arange(0,T*dt,dt)
xs = np.zeros(time.shape)
ys = np.zeros(time.shape)  #2D location
prehist = max(len(K_dc),len(K_dcp))  #pre-histroy length

# %%
###Inference here
    #mixture likelihood
    #separation of speed and anlges
#negative log-likelihood
def nLL_dtheta(THETA, dth,dcp,dc):
    """
    negative log-likelihood objective function for fitting
    THETA includes parameter to be inferred and dth, dcp, dc are from recorded data
    """
    alpha, tau, K, A, B, gamma, K2 = THETA[0], THETA[1], THETA[2], THETA[3], THETA[4:9], THETA[9], THETA[10]
    #alpha and tau for K_dcp parameter, K for VM variance, A for logisic numerator, B for K_dc kernel
    #gamma for mixture of random turn, and K2 for the varaicne of turning VM
    K_dcp = -alpha *np.exp(-K_win/tau)  #sign change due to the way simulated above
    VM = vonmises.pdf((dth-np.dot(dcp,K_dcp))*d2r, K, loc=0, scale=1)/4  #pdf forweather-vaning
    K_dc = np.dot(B,RaisedCosine_basis(len(K_win),5))  #basis function for turning kernel
    P = sigmoid2(A, K_dc, dc)    #logisitc form of turning probability
    gamma = max(0,min(gamma,1))  #it is a ratio in the mixture
    marginalP = np.multiply((1-P), VM) + ((1-gamma)/(2*np.pi) + gamma*vonmises.pdf(dth*d2r, K2, loc=np.pi, scale=1))*P
    nll = -np.sum(np.log(marginalP+1e-9))
    return np.sum(nll)

def nLL_velocity(THETA,vs,dc):
    """
    negative log-likelihood for velocity under a given concentration
    """
    
    return

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
            dC = np.random.randn(len(K_dc))*0.01  #white noise for mapping
            dc_perp = np.random.randn(len(K_dcp))*.01  #white noise for mapping   
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
###fitting part
data_th, data_dcp, data_dc = generate_noise(100)

# %%
#optimize all with less parameters
theta_guess = np.array([50,0.5,100,0.1])  #alpha, tau, K, A
theta_guess = np.concatenate((theta_guess,np.random.randn(5)))  #random weight for basis of Kdc kernel
theta_guess = np.concatenate((theta_guess,np.array([0.5, 10])))  #gamma, K2(sigma)
bnds = ((0,None),(0,None),(0,None),(0,None),(None,None),(None,None),(None,None),(None,None),(None,None),(0,1),(0,None))
res = scipy.optimize.minimize(nLL_dtheta,theta_guess,args=(data_th,data_dcp,data_dc))#,bounds=bnds)#, method='Nelder-Mead')#, bounds=bnds)
#ground truth: 50, 0.5, 100, 0.1, [], 0.5, 10

theta_fit = res.x

# %%
### check kernel forms!!
fit_par = theta_fit[4:9]
recKdc = np.dot(fit_par,RaisedCosine_basis(len(K_dc),len(fit_par)))  #reconstruct Kdc kernel
plt.plot(recKdc,'b',label='K_c_fit',linewidth=3)
plt.plot(K_dc,'b--',label='K_c',linewidth=3)  #compare form with normalized real kernel  ##/np.linalg.norm(K_dc)
recKdcp = theta_fit[0]*np.exp(-K_win/theta_fit[1])
plt.plot(recKdcp,'r',label='K_cp_fit',linewidth=3)
plt.plot(K_dcp,'r--',label='K_cp',linewidth=3)
plt.legend(fontsize=15)

# %%
###check densities
#sigmoid curve
conv_dc1 = np.dot(recKdc,data_dc.T)
pxp = sigmoid2(theta_fit[3],recKdc,data_dc)
plt.plot(conv_dc1,pxp,'o',linewidth=3,label='inferred',color='r',alpha=0.1)
conv_dc = np.dot(K_dc,data_dc.T)
plt.plot(conv_dc, sigmoid2(5*0.023, K_dc,data_dc),'o',linewidth=5,label='true',alpha=0.1)
plt.xlabel('filtered dC',fontsize=20)
plt.ylabel('P(turn)',rotation='horizontal',fontsize=20) 
plt.grid(True)
plt.legend(fontsize=15)

# %%
### check on von Mises density
# turning bias
temp1 = np.array([VM_U(gamma,sigma) for ii in range(1000)])
aa,bb = np.histogram(temp1,bins=50)
#plt.hist(temp1,bins=50,label='true')
plt.bar(bb[:-1],aa/np.sum(aa),align='center',width=0.15,color='k',label='true')
temp2 = np.array([VM_U(theta_fit[9],theta_fit[10]) for ii in range(1000)])
aa,bb = np.histogram(temp2,bins=50)
#plt.hist(temp2,bins=50,alpha=0.5,color='r',label='inferred')
plt.bar(bb[:-1],aa/np.sum(aa),alpha=0.5,align='center',width=0.15,color='grey',label='inferred')
plt.legend(fontsize=20)
plt.xlabel('heading',fontsize=20)
plt.ylabel('pdf',fontsize=20)
plt.xticks((-3.14, 0, 3.14), ('$-\pi$', '$0$', '$+\pi$'), color='k', size=20)
# weather-vaning
plt.figure()
aa,bb = np.histogram((data_th-np.dot(data_dcp,recKdcp))*d2r,bins=200)
plt.bar(bb[:-1],aa/len(data_th),align='edge',width=0.03,color='k',label='true')
rv = vonmises(theta_fit[2])
plt.bar(bb[:-1],rv.pdf(bb[:-1])*np.mean(np.diff(bb)),alpha=0.5,align='center',width=0.03,color='grey',label='inferred')
#plt.axis([-.5,.5,0,0.5])
plt.axis([-0.785,0.785,0,0.23])
plt.legend(fontsize=20)
plt.xlabel('heading',fontsize=20)
plt.ylabel('pdf',fontsize=20)
plt.xticks((-0.785, 0, 0.785), ('$-\pi/4$', '$0$', '$+\pi/4$'), color='k', size=20)
#normalization by bin size???
#checking pdf density
print('sum of histogram:',np.sum(aa/len(data_th)))
print('integrate von Mises:',np.sum(rv.pdf(bb[:-1])*np.mean(np.diff(bb))))