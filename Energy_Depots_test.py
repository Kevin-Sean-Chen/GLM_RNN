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

import matplotlib 
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

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

# %% 
###############################################################################
# %% Active matters
###############################################################################
# %% functions
def F_(v):
    #ff = -g1*v + g2*v**3
    #ff = -v*(1-v)+v**2/(.1+v**2)
    ff = -g1*v + g2*v*(v+aa/np.sqrt(1))*(v+bb/np.sqrt(1)) +0.*(v**2-1)
    return ff
def U_(x):
    q_pot = V0 - V1*(x-x0) - 0.5*V2*(x-x0)**2
    return q_pot
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

# %% dynamics
T = 1000
dt = 0.01
lt, time = len(np.arange(0,T,dt)), np.arange(0,T,dt)
xs,vs, dus = np.zeros(lt), np.zeros(lt), np.zeros(lt)
g1, g2 = .5,.5/3
noise_x, noise_v = 5,10
V0,V1,V2,x0 = 0.,.5,.01,300
aa,bb = -2.,1.
ws = np.zeros(lt)
for tt in range(lt-1):
    dUdx = derivative(U_,xs[tt])   ####check thissssss~~~~~~~
    #dUdx = -dt*(U_(xs[tt+1]) - U_(xs[tt]))
    xs[tt+1] = xs[tt] + dt*vs[tt] + np.sqrt(dt*noise_x)*np.random.randn() + dUdx*0.01
    vs[tt+1] = vs[tt] + dt*( -F_(vs[tt]) + dUdx*1 - ws[tt] ) + np.sqrt(dt*noise_v)*np.random.randn()
    #ws[tt+1] = ws[tt] + dt*(0.08*(vs[tt]+0.7-0.8*ws[tt]))
    dus[tt] = dUdx
    
# %% anaysis
plt.figure()
plt.plot(time,xs)  #trajectory
plt.figure()
plt.plot(dus,vs,'.',alpha=0.01)  #response
plt.figure()
plt.hist(vs,100)   #clustering
acf = autocorr(vs)
win = 5000
plt.figure()
plt.plot(time[:win],acf[:win])   #autocorrelation

# %%
thr = g2/g1
pos = np.where(np.abs(dus)<thr)[0]
plt.figure()
plt.plot(dus[pos],vs[pos],'.',alpha=0.1)  #steering
H = np.histogram2d(dus[pos],vs[pos],bins=100)
cnts,bns = H[0],H[1][:-1]
avg = np.mean(H[0],axis=1)*bns
plt.figure()
plt.plot(bns,avg)

###############################################################################
# %% Effect of noise soucre
###############################################################################
# %%
def Nonlinear(x,thd,gai):
    r = 1/(1+np.exp(-1/gai*(x-thd))) + 0.
    return r
def Environment(x,slp,nis):
    c = x*slp + nis*np.random.randn()
    return c
def Turn(p,a):
    if p>np.random.rand():
        beta = a
    else:
        beta = -a  #change to change!!!
    return beta

# %%
thd, gai, slp, nis = 0, 1, 10, .1
dt, vm, vv = 0.1, 1, .1
N = 500
T = 500
xs = np.zeros((N,T))
for nn in range(N):
    sen_ = 0
    act_ = np.random.randint(0,2)*2-1
    for tt in range(T-1):
        sen = Environment(xs[nn,tt],slp,nis)
        ds = sen-sen_
        sen_ = sen
        nli = Nonlinear(ds,thd,gai)
        act = Turn(nli, act_)
        act_ = act
        v = vm + vv*np.random.randn()
        xs[nn,tt+1] = xs[nn,tt] + act*v*dt

plt.figure()
plt.plot(xs.T)

# %%
def chemotaxis(params,N,T):
    thd, gai, slp, nis = params
    xs = np.zeros((N,T))
    for nn in range(N):
        sen_ = 0
        act_ = np.random.randint(0,2)*2-1
        for tt in range(T-1):
            sen = Environment(xs[nn,tt],slp,nis)
            ds = sen-sen_
            sen_ = sen
            nli = Nonlinear(ds,thd,gai)
            act = Turn(nli,act_)
            act_ = act
            v = vm + vv*np.random.randn()
            xs[nn,tt+1] = xs[nn,tt] + act*v*dt
    return xs[:,-1]

# %%
thd, gai, slp, nis = -0, 1, 10, .1
dt, vm, vv = 0.1, 1, .1
thdS = np.arange(-2,2,0.2)
gaiS = np.logspace(-1,2,10,base=10)
C_avg = np.zeros((len(thdS),len(gaiS)))
C_var = np.zeros((len(thdS),len(gaiS)))
for ii in range(len(thdS)):
    for jj in range(len(gaiS)):
        params = thdS[ii], gaiS[jj], slp, nis
        xf = chemotaxis(params,N,T)
        C_avg[ii,jj] = np.mean(xf)
        C_var[ii,jj] = np.var(xf)
        print(ii,jj)
        
# %%
plt.figure()
plt.imshow(C_var.T,extent=[min(thdS),max(thdS),min(gaiS),max(gaiS)],aspect='auto')
plt.xlabel('threshold',fontsize=40)
plt.ylabel('gain',fontsize=40)


