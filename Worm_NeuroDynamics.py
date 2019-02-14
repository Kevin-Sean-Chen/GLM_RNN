#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:33:04 2019

@author: kschen
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import csv


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
    if typp=='chemical': #'electrical':#
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


###randomly assign E-I neurons
ratioI = 0.
ins = int(ratioI*ns)
selected = np.random.choice(ns, ins)
AA[selected] = -1*AA[selected]

plt.figure(figsize=(20,20))
# AA[AA>0] = 1
plt.imshow(AA[:,:],cmap="hot",interpolation='none')
plt.imshow(AA[:,:],cmap="gray",interpolation='none')

Ak = AA*0
BB = AA.copy()
for ii in range(BB.shape[0]-1,0,-1):
    temp = np.where(BB[ii,:] != 0)
    Ak[ii,np.squeeze(temp)] = 1
    if len(temp[0])==1:
        plt.plot(np.squeeze(temp),ii,'bo')
        plt.hold(True)
    else:
        plt.plot(np.squeeze(temp),np.zeros((np.squeeze(temp)).shape[0])+ii,'b.')
        plt.hold(True)

        
nn = AA.shape[0]
T = 100
dt = 0.01
time = np.arange(0,T,dt)

def NLf(vs,slp,rect):
    vs_NL = np.exp(vs)
    vs_NL[vs_NL>rect] = rect
    return vs_NL

slp = 1.
rect = 10
tau = 20
vs = np.zeros((nn,len(time)))
vs[:,0] = np.random.randn(nn)
for t in range(0,len(time)-1):
#     vs[:,t+1] = vs[:,t] + dt*(-vs[:,t]/tau + NLf(vs[:,t] @ AA,slp,rect) + np.random.randn(ns)*0.5)
    vs[:,t+1] = vs[:,t] + dt*(-vs[:,t] + (vs[:,t] @ AA)/tau + np.random.randn(ns)*0.5)

   
plt.plot(time,vs.T);
plt.xlabel('time',fontsize=20)
plt.ylabel('activity',fontsize=20)