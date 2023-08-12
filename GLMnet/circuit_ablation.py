# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 01:55:48 2023

@author: kevin
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy as sp
from scipy.optimize import minimize

import matplotlib 
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25)

# %% circuit model setup
N = 5  # number of interneurons (AIB, AIY, AIZ, AIA, RIA; downstream from AWC)
B = 2  # number of motor readout (WV and BRW)
C = 5  # number of ablation conditions

def NL(x):
    return np.maximum(0, x)

# %% true model
Wim = np.random.randn(N,B)  # weights from interneuron to motor
Vi = np.random.rand(N)  # voltage of interneurons
Vm = NL(Wim.T @ Vi)  # voltage of motor readout

# %% ablation information
M_nc = np.zeros((N,C))  # matrix for neuron ablations
M_bc = np.zeros((B,C))  # matrix for behavior given ablation
for cc in range(C):
    Vab = Vi*1
    Vab[cc] = 0
    M_nc[:,cc] = Vab
    M_bc[:,cc] = NL(Wim.T @ Vab)

Winf = np.linalg.solve(M_nc.T,M_bc.T)
# %% plot results
plt.figure()
plt.plot(Wim,'k-o',label='true')
plt.plot(Winf,'b-o',linewidth=20,alpha=0.5,label='inferred')
plt.legend(fontsize=30)
