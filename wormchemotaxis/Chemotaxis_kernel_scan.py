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
    perp_dir = np.array([-dxy[1], dxy[0]])
    perp_dir = perp_dir/np.linalg.norm(perp_dir)
    perp_dC = gradient(C0, xx+perp_dir[0], yy+perp_dir[1]) - gradient(C0, xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

#chemotaxis strategy parameter
def generate_chemotaxis(Ww,Wp,rep):
    """
    generate chemotaxis behavior trajectories given weights on reversal and weathervaning kernels
    measure distance to target and sum of odor
    """
    K_win = np.linspace(0,6,6/0.6)
    #K_dc = Wp*(temporal_kernel(4.,K_win))+.0  #random-turning kernel (biphasic form)
    K_dc = 30*(temporal_kernel(Wp,K_win))+.0
    K_dc = K_dc - np.mean(K_dc)  #zero-mean kernel for stationary solution
    #wv_win = 0.8
    #K_dcp = Ww*np.exp(-K_win/wv_win)  #weathervaning kernel (exponential form)
    K_dcp = 30*np.exp(-K_win/Ww)
    
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

    #####
    #Generate trajeectories
    #####
    final_dis = []
    accu_dc = []
    accu_dcp = []
    for ii in range(rep):
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
            dC = np.array([gradient(C0, xs[t-past],ys[t-past]) for past in range(0,len(K_dc))])
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

        ###record
        final_dis.append(abs(xs[-1]-dis2targ))  #fistance to target
        accu_dc.append(np.sum(dcs))  #experienced dC
        accu_dcp.append(np.sum(dcps))  #experienced dC_p
        
        #plt.plot(xs,ys)     
    return final_dis, accu_dc, accu_dcp


### parameters and environemnt
#gradient environment
dis2targ = 50
C0 = 0.2
D = 0.000015
duT = 60*60*3
d = 0.18

#other behavior parameters
K = 1  #covariance of weathervane
T = 5000
dt = 0.6  #seconds
v_m = 0.12  #mm/s
v_s = 0.01  #std of speed
time = np.arange(0,T*dt,dt)

###scanning parameters
rep = 30
#Wws = np.arange(1,55,5)
Wws = np.arange(0.1,2,0.2)
#Wps = np.arange(1,55,5)
Wps = np.arange(1,11,1)
all_final = []
all_dc = []
all_dcp = []
iii = 0
for Ww in Wws:
    for Wp in Wps:
        final_dis, accu_dc, accu_dcp = generate_chemotaxis(Ww,Wp,rep)
        all_final.append(final_dis)
        all_dc.append(accu_dc)
        all_dcp.append(accu_dcp)
        iii = iii+1
        print(iii)
        
###analysis code
temp = [np.std(ii) for ii in all_final]
temp2 = np.reshape(temp,(len(Wps),len(Wws)))
plt.imshow(temp2,origin='lower')
plt.xticks(range(len(Wws)),Wws)
plt.yticks(range(len(Wps)),Wps)
plt.xlabel('W_turn')
plt.ylabel('W_heading')
plt.colorbar()
