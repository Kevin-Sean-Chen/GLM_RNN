# -*- coding: utf-8 -*-
"""
Created on Mon May  4 18:49:14 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import dotmap as DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

#%matplotlib qt5
    
# %% Circuit class
class Circuit(object):
    def __init__(self, T, N, J, A, n_pump = 0, dt = 0.1):
        time = np.arange(0,T,dt)
        lt = len(time)
        self.dt = dt
        self.T = T
        self.time = time
        self.N = N
        self.lt = lt
        self.J = J
        self.n_pump = n_pump
        self.A = A
        self.eps = 10**-15
    
    def LN(self, x):
        """
        logistic nonlinearity
        """
        ln = 1/(1+np.exp(-x*1.+self.eps))  #np.tanh(x)  #   #logsitic
        return ln  #np.random.poisson(ln) #ln  #Poinsson emission

    def spiking(self, ll, dt):
        """
        Given Poisson rate (spk per second) and time steps dt return binary process with the probability
        """
        N = len(ll)
        spike = np.random.rand(N) < ll*dt  #for Bernouli process
        return spike
    
    def simulation(self):
        """
        Neural dynamics of the circuit model
        """
        #settings
        dt, lt, N = self.dt, self.lt, self.N
        x = np.zeros((N,lt))  #voltage
        spk = np.zeros_like(x)  #spikes
        syn = np.zeros_like(x)  #synaptic efficacy
        rate = np.zeros_like(x)  #spike rate
        x[:,0] = np.random.randn(N)*1
        syn[:,0] = np.random.rand(N)
#        stim = np.random.randn(lt)*self.A  #np.random.randn(N,lt)*20.  #common random stimuli
        ###
        stim = self.stimulate(self.n_pump, self.A)
        ###
        #biophysical parameters
        J = self.J.T#connectivity matrix
        noise = 1.  #noise strength
        taum = 5  #5 ms
        taus = 50  #50 ms
        E = 1
        #iterations for neural dynamics
        for tt in range(0,lt-1):
            x[:,tt+1] = x[:,tt] + dt/taum*( -x[:,tt] + (np.matmul(J,self.LN(syn[:,tt]*x[:,tt]))) + stim[:,tt] + noise*np.random.randn(N)*np.sqrt(dt))
            spk[:,tt+1] = self.spiking(self.LN(x[:,tt+1]),dt)
            rate[:,tt+1] = self.LN(x[:,tt+1])
            syn[:,tt+1] = 1#syn[:,tt] + dt*( (1-syn[:,tt])/taus - syn[:,tt]*E*spk[:,tt] )
        return x, spk, rate, stim
    
    def stimulate(self,n_pump, A):
        """
        Implement pump-probe stimuluation protocol here
        """
        stim = np.zeros((self.N,self.lt))
        ###general noise driven dynamics
        if n_pump=='noise':
            stim_temp = np.random.randn(self.lt)*A
            stim = np.repeat(stim_temp,repeats=self.N).reshape(self.lt,self.N).T
        ###pump-probe use
        else:
            pos = np.where((self.time>500) & (self.time<510))[0]
            stim[n_pump,pos] = A
            pos = np.where((self.time>700) & (self.time<710))[0]
            stim[n_pump,pos] = A
            pos = np.where((self.time>900) & (self.time<910))[0]
            stim[n_pump,pos] = A
        return stim
    
# %% simulation and analysis
J = np.array([[6.8, -2.5, -2],\
              [-3, 7., -2],\
              [-2.3, -2.5, 4.1]])
C_ = Circuit(T=5000, N=3, J=J, A=20, n_pump = 0, dt = 0.1)
xx,ss,rr,stim = C_.simulation()

# %%
# %% circuit connections
def connections(typ):
    if typ == 'chain':
        J = np.array([[0, 0.5, 0],\
                      [0, 0, 0.5],\
                      [0, 0, 0]])
    if typ == 'loop':
        J = np.array([[0, 1/3, 0],\
                      [0, 0, 1/3],\
                      [1/3, 0, 0]])
    if typ == 'inhibit':
        J = np.array([[0, 2/3, 2/3],\
                      [0, 0, -1/3],\
                      [0, 0, 0]])
    if typ == 'DAG':
        J = np.array([[0, 1/3, 1/3],\
                      [0 ,0 , 1/3],\
                      [0, 0, 0]])
    if typ == 'test':
#        J = np.array([[0, -1/3, 1/3],\
#                      [-1/3 ,0 , -1/3],\
#                      [-1/3, 1/3, 0]])
        J = np.array([[6.8, -2.5, -2],\
              [-3, 7., -2],\
              [-2.3, -2.5, 4.1]])
    return J
# %% simulation
J = connections('test')
C_ = Circuit(T=1000, N=3, J=J, A=30, n_pump = 'noise', dt = 0.1)
xx,ss,rr,stim = C_.simulation()

# %%
plt.figure()
plt.subplot(411)
plt.imshow(xx, aspect='auto');
plt.subplot(412)
plt.imshow(ss, aspect='auto');
plt.subplot(413)
plt.imshow(rr, aspect='auto');
plt.xlim([0,C_.lt])
plt.subplot(414)
plt.plot(C_.time, rr.T);
plt.xlim([0,C_.time[-1]])

# %% GLM inference
# %% generative model
J = connections('DAG')
C_ = Circuit(T=1000, N=3, J=J, A=20, n_pump = 'noise', dt = 0.1)
xx,ss,rr,stim = C_.simulation()

# %% inference method (single)
nneuron = 0
pad = 100  #window for kernel
nbasis = 7  #number of basis
couple = 1  #wether or not coupling cells considered
Y = np.squeeze(rr[nneuron,:])  #spike train of interest
Ks = (np.fliplr(basis_function1(pad,nbasis).T).T).T  #basis function used for kernel approximation
stimulus = stim[nneuron,:][:,None]  #same stimulus for all neurons
X = build_convolved_matrix(stimulus, rr.T, Ks, couple)  #design matrix with features projected onto basis functions
###pyGLMnet function with optimal parameters
glm = GLMCV(distr="binomial", tol=1e-5, eta=1.0,
            score_metric="deviance",
            alpha=0., learning_rate=1e-6, max_iter=1000, cv=3, verbose=True)  #important to have v slow learning_rate
glm.fit(X, Y)

# %% direct simulation
yhat = simulate_glm('binomial', glm.beta0_, glm.beta_, X)  #simulate spike rate given the firring results
plt.figure()
plt.plot(Y*1.)  #ground truth
plt.plot(yhat,'--')

# %%reconstruct kernel
theta = glm.beta_
dc_ = theta[0]
theta_ = theta[1:]
if couple == 1:
    theta_ = theta_.reshape(N+1,nbasis).T  #nbasis times (stimulus + N neurons)
    allKs = np.array([theta_[:,kk] @ Ks for kk in range(N+1)])
elif couple == 0:
    allKs = Ks.T @ theta_

plt.figure()
plt.plot(allKs.T)
 
    