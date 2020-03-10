# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:41:26 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

#%matplotlib qt5
# %% functions
def Energy_potential(r):
    """
    Similar to "value"?
    """
    x,y = r
    U = a_/2*(x**2+y**2)  #simple parabolic potential
    return U

def State_energy(r,q0):
    """
    Similar to "reward"?
    """
    x,y = r
    R = 1  #designing an arbitrary landscape with radius R and center C
    C = np.array([1,1])*0.1
    q = np.exp(np.linalg.norm(r-C)**2/R)
    if np.linalg.norm(r-C)**2 <= R:
        #q = q0
        if np.random.rand()>0.5:
            q = q0
        else:
            q = 0
    else:
        q = 0
    return q

def Kinetic_energy(v,d2):
    """
    Like an action "cost"
    """
    d = d2*np.linalg.norm(v)**2
    return d

def derivative(f,a,method='central',h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h            
    '''
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")
    
# %% dynamics
###setting
dt = 0.1
T = 1000
time = np.arange(0,T,dt)
lt = len(time)
rs = np.zeros((2,lt))
vs = np.zeros((2,lt))
es = np.zeros(lt)
#rs[:,0] = np.random.randn(2)
#vs[:,0] = np.random.randn(2)
#es[0] = np.random.randn()

###parameters
a_ = 2
gamma0 = 0.2
d2 = 1.
c = 0.1
S = 0.1
q0 = 1.
m = 1

for tt in range(0,lt-1):
    rs[:,tt+1] = rs[:,tt] + dt*(vs[:,tt])
    #vs[:,tt+1] = vs[:,tt] + dt*(1/m)*(-gamma0*vs[:,tt] - derivative(Energy_potential,rs[:,tt],method='central',h=0.01) + np.sqrt(dt)*S*np.random.randn(2))
    vs[:,tt+1] = vs[:,tt] + dt*(1/m)*(-gamma0*vs[:,tt] - (np.array([a_,0]*rs[:,tt] + np.array([0,a_])*rs[:,tt])) + np.sqrt(dt)*S*np.random.randn(2))
    es[tt+1] = es[tt] + dt*(State_energy(rs[:,tt],q0) - c*es[tt] - Kinetic_energy(vs[:,tt],d2)*es[tt])

plt.plot(rs[0,:], rs[1,:])
plt.plot(rs[0,0], rs[1,0],'o')