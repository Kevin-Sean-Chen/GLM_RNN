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

#Hybrid Gaussian process for angle time series
def d_theta(alpha, dc_perp, K, w, dC):
    '''
    Return change in theta angle for each step
    Input with alpha for weighting, dc for orthogonal concentration difference, and K covariance in weathervaning
    W as the weighting/kernel on concentration in the signoidal function for tumbling rate 
    '''
    wv = alpha*dc_perp + K*np.random.randn()  #weathervaning strategy
    #P_event = 0.023/(0.4 + np.exp(40*dC/dt)) + 0.003  #sigmoidal function with parameters w
    P_event = 5*0.023/(1 + np.exp(40*dC/dt))  #less parameter version
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
<<<<<<< HEAD
    ### for linear gradient
    #concentration = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50))  #depends on diffusion conditions
    ### for Gaussian gradient
    dist = np.linalg.norm(np.array([x,y])-np.array([dis2targ,0]))
    concentration = C0/(4*np.pi*d*D*duT)*np.exp(-dist**2/(400*D*duT*50))
=======
    #linear gradient
    concentration = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50))  #depends on diffusion conditions
    #Gaussian gradient
    dist = np.linalg.norm(np.array([x,y])-np.array([dis2targ,0]))
    concentration = C0/(4*np.pi*d*D*duT)*np.exp(-(dist)**2/(400*D*duT*50)) 
>>>>>>> 5ed18d182680807767c7bcb08f0dd0014283b6c6
    return concentration

#measure for concentration difference for weathervane
def dc_measure(dxy,xx,yy):
    perp_dir = np.array([-dxy[1], dxy[0]])
    perp_dir = perp_dir/np.linalg.norm(perp_dir)
    perp_dC = gradient(C0, xx+perp_dir[0], yy+perp_dir[1]) - gradient(C0, xx-perp_dir[0], yy-perp_dir[1])
    return perp_dC

#gradient environment
dis2targ = 50
C0 = 0.2
D = 0.000015
duT = 60*60*1
d = 0.18

#chemotaxis strategy parameter
alpha = 30  #strength of OU forcing
K = 10  #covariance of weathervane
w = 0  #logistic parameter (default for now)
T = 1500
dt = 0.6  #seconds
v_m = 0.12  #mm/s
v_s = 0.01  #std of speed
time = np.arange(0,T*dt,dt)
xs = np.zeros(time.shape)
ys = np.zeros(time.shape)  #2D location
xs[0] = np.random.randn()
ys[0] = np.random.randn()
ths = np.zeros(time.shape)  #agle with 1,0
ths[0] = np.random.randn()
dxy = np.random.randn(2)

### with turning (Brownian-like tragectories)
#for t in range(1,len(time)):
#    
#    #concentration = gradient(C0,xs[t-1],ys[t-1])
#    dC = gradient(C0, xs[t-1],ys[t-1]) - gradient(C0, xs[t-2],ys[t-2])
#    dc_perp = dc_measure(dxy,xs[t-1],ys[t-1])      
#    dth = d_theta(alpha, -dc_perp, K, 0, dC)
#    ths[t] = ths[t-1] + dth*dt
#    
#    e1 = np.array([1,0])
#    vec = np.array([xs[t-1],ys[t-1]])
#    theta = math.acos(np.clip(np.dot(vec,e1)/np.linalg.norm(vec)/np.linalg.norm(e1), -1, 1)) #current orienation relative to (1,0)
#
#    vv = v_m + v_s*np.random.randn()
#    dd = np.array([vv*np.sin(ths[t]*np.pi/180), vv*np.cos(ths[t]*np.pi/180)])  #displacement
#    c, s = np.cos(theta), np.sin(theta)
#    R = np.array(((c,s), (-s, c)))  #rotation matrix, changing coordinates
#    dxy = np.dot(R,dd)
#    
#    xs[t] = xs[t-1] + dxy[0]*dt
#    ys[t] = ys[t-1] + dxy[1]*dt
#
##plt.plot(ths)
#plt.figure()
#plt.plot(xs,ys)
#plt.figure()
#x = np.arange(np.min(xs),np.max(xs),1)
#xx_grad = C0/(4*np.pi*d*D*duT)*np.exp(-(x-dis2targ)**2/(400*D*duT*50))
#plt.imshow(np.expand_dims(xx_grad,axis=1).T,extent=[np.min(xs),np.max(xs),np.min(ys),np.max(ys)])
#plt.hold(True)
#plt.plot(xs,ys,'white')

#####
#Generate trajeectories
#####
all_dc_p = []
all_dc = []
all_th = []
for ii in range(30):
    xs = np.zeros(time.shape)
    ys = np.zeros(time.shape)  #2D location
    xs[0] = np.random.randn()*0.1
    ys[0] = np.random.randn()*0.1
    ths = np.zeros(time.shape)  #agle with 1,0
    ths[0] = np.random.rand()*0
    dcs = np.zeros(time.shape)
    dcps = np.zeros(time.shape)
    dths = np.zeros(time.shape)
    for t in range(1,len(time)):
        
        dC = gradient(C0, xs[t-1],ys[t-1]) - gradient(C0, xs[t-2],ys[t-2])
        dc_perp = dc_measure(dxy,xs[t-1],ys[t-1])      
        dth = d_theta(alpha, -dc_perp, K, 0, dC)
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
        
    plt.plot(xs,ys)
    #plt.hold(True)

###ALL DATA HERE~~
data_th = np.array(all_th).reshape(-1)
data_dcp = np.array(all_dc_p).reshape(-1)
data_dc = np.array(all_dc).reshape(-1)

# %%
#####
#Inference for chemotactic strategy
#####

#von Mises distribution test
d2r = np.pi/180
vm_par = vonmises.fit((data_th-alpha*data_dcp)*d2r, scale=1)
plt.hist(data_th*d2r,bins=100,normed=True);
#plt.hold(True)
xx = np.linspace(np.min(data_th*d2r),np.max(data_th*d2r),100)
rv = vonmises(vm_par[0])
plt.plot(xx, rv.pdf(xx),linewidth=3)

#negative log-likelihood
def nLL(THETA, dth,dcp,dc):
    #a_, k_, A_, B_, C_, D_ = THETA  #inferred paramter
    a_,k_,A_,B_ = THETA
    #P = sigmoid(A_, B_, C_, D_, dcp)
    P = sigmoid2(A_,B_,dc)
    #VM = np.exp(k_*np.cos((dth-a_*dcp)*d2r)) / (2*np.pi*iv(0,k_))#von Mises distribution
    #vm_par = vonmises.fit((dth-a_*dcp)*d2r, scale=1)
    rv = vonmises(k_)#(vm_par[0])
    VM = rv.pdf((dth-a_*dcp)*d2r)
    marginalP = np.multiply((1-P), VM) + (1/(2*np.pi))*P
    nll = -np.sum(np.log(marginalP+1e-7))#, axis=1)
    #fst = np.einsum('ij,ij->i', 1-P, VM)
    #snd = np.sum(1/np.pi*P, axis=1)
    return np.sum(nll)

rv = vonmises(K)
VM = rv.pdf((data_th-alpha*data_dcp)*d2r)
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
    y = a / (1 + np.exp(b*x))
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

#optimize all with less parameters
theta_guess = 1,1,0.1,100  #,10,0.001  #a_, k_, A_, B_
###Ground Truth: 25,5,0.023,0.4,40,0.003
res = scipy.optimize.minimize(nLL,theta_guess,args=(data_th,data_dcp,data_dc),method='Nelder-Mead')
                              #,bounds = ((0,None),(0,None),(None,None),(None,None)))
theta_fit = res.x
##optimize logistic
#theta_guess = 1,0.5  #a_, k_, A_, B_, C_, D_ 
####Ground Truth: 25,5,0.023,0.4,40,0.003
#res = scipy.optimize.minimize(nLL2,theta_guess,args=(VM,data_th,data_dcp,data_dc),bounds = ((0,None),(0,None)))
#theta_fit = res.x

# %%
###check on von Mises density
#plt.hist((data_th-alpha*data_dcp)*d2r,bins=100,normed=True,color='r');
aa,bb = np.histogram((data_th-alpha*data_dcp)*d2r,bins=200)
plt.bar(bb[:-1],aa/len(data_th),align='edge',width=0.03)
#plt.hold(True)
rv = vonmises(res.x[1])
#plt.scatter((data_th-alpha*data_dcp)*d2r,rv.pdf((data_th-alpha*data_dcp)*d2r),s=1,marker='.')
plt.bar(bb[:-1],rv.pdf(bb[:-1])*np.mean(np.diff(bb)),alpha=0.5,align='center',width=0.03,color='r')
plt.axis([-1,1,0,0.2])
#normalization by bin size???
#checking pdf density
print('sum of histogram:',np.sum(aa/len(data_th)))
print('integrate von Mises:',np.sum(rv.pdf(bb[:-1])*np.mean(np.diff(bb))))

###check on logistic fitting
plt.figure()
xp = np.linspace(-0.5, 0.5, 1000)
pxp=sigmoid2(res.x[2],res.x[3],xp)
#pxp=sigmoid1(res.x[2],res.x[3],res.x[4],res.x[5],xp)
plt.plot(xp,pxp,'-',linewidth=3,label='fit')
#plt.hold(True)
plt.plot(xp,sigmoid2(5*0.023, 140/0.6,xp),linewidth=5,label='ground-truth',alpha=0.8)
#plt.plot(xp,sigmoid1(0.023,0.4,140/0.6,0.003,xp),linewidth=3,label='ground-truth')
plt.xlabel('x')
plt.ylabel('y',rotation='horizontal') 
plt.grid(True)
plt.legend()

### step-wise fitting
#aa,bb = np.histogram(data_dc,bins=10)
#dC_bin = []
#for i in range(0,len(bb)-1):
#    pos = np.where((data_dc>bb[i]) & (data_dc<=bb[i+1]))[0]
#    dC_bin.append(pos)
#def nLL3(THETA, dth,dcp):
#    a_, k_, p_ = THETA  #inferred paramter
#    rv = vonmises(k_)
#    VM = rv.pdf((dth-a_*dcp)*d2r)
#    marginalP = np.multiply((1-p_), VM) + (1/(2*np.pi))*p_
#    nll = -np.sum(np.log(marginalP+1e-7))
#    return np.sum(nll)
#theta_guess = np.array([10,10,0.1])
#res = scipy.optimize.minimize(nLL3,theta_guess,args=(data_th,data_dcp),bounds = ((None, None), (0,None),(0,1)))
#theta_fit = res.x