# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:48:45 2021

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
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

# %% Circuit dynamics
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% Circuit class
class Circuit_dynamics(object):
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
        #ln = np.exp(x)
        ln = 1/(1+np.exp(-x*1.+self.eps))  #np.tanh(x)  #   #logsitic
        return ln  #np.random.poisson(ln) #ln  #Poinsson emission

    def spiking(self, ll, dt):
        """
        Given Poisson rate (spk per second) and time steps dt return binary process with the probability
        """
        N = len(ll)
        spike = np.random.rand(N) < ll*dt  #for Bernouli process
        #spike = np.random.poisson(ll*dt)
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
        noise = .1  #noise strength
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
        elif n_pump=='sine':
            stim_temp = np.sin(np.arange(self.lt)/A)
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
    
    def flipkernel(self,k):
        """
        flipping kernel to resolve temporal direction
        """
        return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
    def GLM_NL(self, x):
        nl = np.exp(x)
        #nl = 1/(1+np.exp(-x*1.+self.eps))
        return nl
    
    def coupled_GLM(self, kernels, mus):
        """
        Ground truth set of coupled GLMS to produce spike trains
        kernels should be (nb+1)xpad
        return spiking probability, spikes, stimuli, and true kernels
        """
        #load dimension and stimuli
        dt, lt, N = self.dt, self.lt, self.N
        stim = self.stimulate(self.n_pump, self.A)
        _,_,pad = kernels.shape  #N x (N+1) x pad
        pad = pad #-1
        
        #kernels has dimension:  N x (N+1) x pad
        Ks = self.flipkernel(kernels[:,0,:].T).T  #stimulu kernel
        Hs = self.flipkernel(kernels[:,1:,:].T).T  #history kernel
        mus = mus #kernels[:,0,0] #baseline
        
        #record GLM network spikes
        spks = np.zeros((N,lt))  #recording all spikes
        Psks = np.zeros((N,lt))  #recording all spiking probability
        for tt in range(pad,lt):
            Psks[:,tt] = self.GLM_NL( np.sum(Ks*stim[:,tt-pad:tt],axis=1) + mus + \
                np.einsum('ijk,jk->i',  Hs, spks[:,tt-pad:tt]) )
            spks[:,tt] = self.spiking(Psks[:,tt], dt)
        
        return Psks, spks, stim, kernels
    
    
    def simulation_pattern(self,model):
        """
        ...
        """
        #time course and network
        dt, lt, N = self.dt, self.lt, self.N

        if model=='bistable':
            ### Bistable ###
            Ithresh = np.array([5, 5]);
            W = np.array([[1.1, -0.15],
                          [-0.15, 1.1]])  #weight matrix
            rinit1 = np.array([50, 55])
            Iapp1 = np.array([ 0, 30])
            Iapp2 = np.array([30, 0])
        elif model=='line':
            ### Line attractor ###
            Ithresh = np.array([-20, -20]);
            W = np.array([[0.7, -0.3],
                          [-0.3, 0.7]])  #weight matrix
            rinit1 = np.array([30, 75])
            Iapp1 = np.array([ 1, 0])
            Iapp2 = np.array([2, 0])
        elif model=='oscillation':
            ### Oscillation attractor ###
            Ithresh = np.array([8, 20]);
            W = np.array([[2.2, -1.3],
                          [1.2, -0.1]])  #weight matrix
            rinit1 = np.array([80, 0])
            Iapp1 = np.array([ 0, 0])
            Iapp2 = np.array([-10, 0])
            
        #initialization
        r = np.zeros((N,lt))    # array of rate of each cell as a function of time
        rmax = 100;             # maximum firing rate
        tau = 0.01;             # base time constant for changes of rate
        
        #stimuli
        Iapp = np.zeros((N,lt))      # Array of time-dependent and unit-dependent current
        Ion1 = 1;                    # Time to switch on
        Ion2 = 2;                    # Time to switch on
        Idur = 0.1;                  # Duration of current
        
        non1 = round(Ion1/dt)            # Time step to switch on current
        noff1 = round((Ion1+Idur)/dt)    # Time step to switch off current
        non2 = round(Ion2/dt)            # Time step to switch on current
        noff2 = round((Ion2+Idur)/dt)    # Time step to switch off current
        
        Iapp[:,non1:noff1] = np.einsum('ij,jk->jk',Iapp1[None,:],np.ones((2,noff1-non1)))
        Iapp[:,non2:noff2] = np.einsum('ij,jk->jk',Iapp2[None,:],np.ones((2,noff2-non2)))
        
        r[:,0] = rinit1                # Initialize firing rate
        
        #spiking network
        tau_m = 0.01  #membrane time constant  #0.01
        tau_r = 0.05  #synaptic rise time scale  #0.05
        tau_d = 0.01   #synaptic decay time  #0.01
        v_the = 9.5    #spik threshold  #13  #7.5  #9.5
        v_res = -1     #reset potential after spiking  #-10 #-1 #-1
        lamb = 35    #factor from rate to spiking networks  #140  #70  #35
        vm = np.zeros((N,lt))  #for membrane potential
        rs = np.zeros((N,lt))  #for spike rate
        ss = np.zeros((N,lt))  #for synaptic input
        spk = np.zeros((N,lt)) #for spikes
        for tt in range(1,lt):
            #neural dynamics
            vm[:,tt] = vm[:,tt-1] + dt*(1/tau_m)*(-vm[:,tt-1] + lamb*W @ rs[:,tt-1] + Iapp[:,tt-1]*5+10) #,embrane potential
            rs[:,tt] = rs[:,tt-1] + dt*(-rs[:,tt-1]/tau_d + ss[:,tt-1])  #spike rate
            ss[:,tt] = ss[:,tt-1] + dt*(-ss[:,tt-1]/tau_r + 1/(tau_d*tau_r)*spk[:,tt-1])  #synaptic input
            #spiking process
            poss = np.where(vm[:,tt]>v_the)[0]  #recording spikes
            if len(poss)>0:
                spk[poss,tt] = 1
            posr = np.where(spk[:,tt-1]>0)[0]  #if spiked
            Vmask = np.ones(N)
            if len(posr)>0:
                vm[posr,tt] = v_res  #reseting spiked neurons
                spk[posr,tt] = 0
        
        return vm, spk, rs, Iapp


# %% GLM inference
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class GLM_inference(object):
    #global wl
    def __init__(self, Y, X, dt, pad, nb):
        self.Y = Y
        self.X = X
        self.dt = dt
        self.pad = pad
        self.nb = nb
        self.wl = 0 #for final calculation
        self.eps = 10**-15
        
    def basis_function(self):
        """
        Raised cosine basis function to tile the time course of the response kernel
        nkbins of time points in the kernel and nBases for the number of basis functions
        """
        nBases = self.nb
        nkbins = self.pad
        #nkbins = 10 #binfun(duration); # number of bins for the basis functions
        ttb = np.tile(np.log(np.arange(0,nkbins)+1)/np.log(1.4),(nBases,1))  #take log for nonlinear time
        dbcenter = nkbins / (nBases+int(nkbins/3)) # spacing between bumps
        width = 5.*dbcenter # width of each bump
        bcenters = 1.*dbcenter + dbcenter*np.arange(0,nBases)  # location of each bump centers
        def bfun(x,period):
            return (abs(x/period)<0.5)*(np.cos(x*2*np.pi/period)*.5+.5)  #raise-cosine function formula
        temp = ttb - np.tile(bcenters,(nkbins,1)).T
        BBstm = [bfun(xx,width) for xx in temp] 
        basis = np.array(BBstm).T
        #plt.plot(np.array(BBstm).T)
        return basis
    
    def flipkernel(self,k):
        """
        flipping kernel to resolve temporal direction
        """
        return np.squeeze(np.fliplr(k[None,:])) ###important for temporal causality!!!??
    
    
    def design_matrix(self, idd, cp_mode, basis_set=None):
        """
        idd:      int for neuron id
        stim:     TxN stimuli
        spk:      TxN spiking pattern
        pad:      int for kernel width, or number of weight parameters
        cp_mode:  binary indication for coupling or independent model
        basis_set:DxB, where D is the kernel window and B is the number of basis used
        """
        spk = self.Y.T
        stim = self.X.T
        pad = self.pad
        
        T, N = spk.shape  #time and number of neurons
        xx = stim[:,idd] #for input stimuli
        D = pad  #pad for D length of kernel
        y = spk  #spiking patterns
        #y = spk[:,idd] #for self spike
        #other_id = np.where(np.arange(N)!=idd)
        #couple = spk[:,other_id]
        
        if basis_set is None:
            if cp_mode==0:
                X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:]) #make design matrix
                X = np.concatenate((np.ones([T,1]),X),axis=1)  #concatenate with constant offset
            elif cp_mode==1:
                X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
                for nn in range(N):
                    yi = y[:,nn]
                    Xi = sp.linalg.hankel(np.append(np.zeros(D-2),yi[:T-D+2]),yi[T-D+1:])  #add spiking history
                    X = np.concatenate((X,Xi),axis=1)
                X = np.concatenate((np.ones([T,1]),X),axis=1)
        else:
            basis = self.flipkernel(basis_set[:-1,:])  #the right temporal order here!
            if cp_mode==0:
                X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
                X = X @ basis  #project to basis set
                X = np.concatenate((np.ones([T,1]),X),axis=1)
            elif cp_mode==1:
                X = sp.linalg.hankel(np.append(np.zeros(D-2),xx[:T-D+2]),xx[T-D+1:])
                X = X @ basis
                for nn in range(N):
                    yi = y[:,nn]
                    Xi = sp.linalg.hankel(np.append(np.zeros(D-2),yi[:T-D+2]),yi[T-D+1:])
                    Xi = Xi @ basis
                    X = np.concatenate((X,Xi),axis=1)
                X = np.concatenate((np.ones([T,1]),X),axis=1)      
            
        y = spk[:,idd]
        self.xl = X.shape[1]  #length of total weight variables
        return y, X
    
    def GLM_NL(self, x):
        nl = np.exp(x)
        #nl = 1/(1+np.exp(-x*1.+self.eps))
        return nl 
    
    def Poisson_log_likelihood(self, w, Y, X, f=np.exp, Cinv=None):
        """
        Poisson GLM log likelihood.
        f is exponential by default.
        """
        f = self.GLM_NL
        dt = self.dt
        # if no prior given, set it to zeros
        if Cinv is None:
            Cinv = np.zeros([np.shape(w)[0],np.shape(w)[0]])
    
        # evaluate log likelihood and gradient
        ll = np.sum(Y * np.log(f(X@w)+self.eps) - f(X@w)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt)) + 0.5*w.T@Cinv@w
        #ll = np.sum(Y * np.log(f(X@w)+self.eps) - (1-Y) * np.log(1-f(X@w)+self.eps) )
        #- sp.special.gammaln(Y+1) + Y*np.log(dt)) + 0.5*w.T@Cinv@w
    
        # return ll
        return ll
    
    
    def MAP_inerence(self, y, X, prior_w=None):
        
        D = X.shape[1]
        # prior
        if prior_w is None:
            lambda_ridge = np.power(2.0,4)
            lambda_ridge = 0.0
            Cinv = lambda_ridge*np.eye(D)
            Cinv[0,0] = 0.0 # no prior on bias
        else:
            Cinv = prior_w
        # fit with MAP
        res = sp.optimize.minimize(lambda w: -self.Poisson_log_likelihood(w,y,X,np.exp,Cinv), np.zeros([D,]),method='L-BFGS-B', tol=1e-4,options={'disp': True})
        #w_map = res.x
        
        return res
    
    def MAP_decoding(w_map, xx, prior_x):
        
        """
        Poisson GLM log likelihood.
        f is exponential by default.
        """
        # if no prior given, set it to zeros
        if Cinv is None:
            Cinv = np.zeros([np.shape(w_map)[0],np.shape(w_map)[0]])
        
        # make design matrix X
        D = len(w_map)
        X = make_designX(xx,D)
        # evaluate log likelihood and gradient
        ll = np.sum(Y * np.log(f(X @ w_map)) - f(X @ w_map)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt)) \
        + 0.5*w_map.T @ Cinv @ w_map + 0.5*(xx-mu_x)[:,None].T @ (xx-mu_x)[:,None]*(sig_x)**-1
    
        # return ll
        #return ll
        
        return X_hat
    
    def recover_kernel(self, basis, w_map):
        B = self.nb
        base = w_map[0]
        Ks = np.reshape(w_map[1:],[int(len(w_map[1:])/B),B])
        Ks = Ks @ basis.T
        return Ks, base

    def spiking(self, ll, dt):
        """
        Given Poisson rate (spk per second) and time steps dt return binary process with the probability
        """
        N = len(ll)
        spike = np.random.rand(N) < ll*dt  #for Bernouli process
        #spike = np.random.poisson(ll*dt)
        return spike
    
    def Poisson_GLM(self, w, X):
        """
        w:  D length weight vector 
        X:  TxD design matrix
        dt: time steps for a bin
        """
        fx = self.GLM_NL(X @ w)
        #np.exp(X @ w)
        Y = self.spiking(fx,self.dt)
        #np.random.poisson(fx*self.dt)
        return Y, fx
    
    def use_pyGLM_inference():
        return glm
    
# %% test analysis
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% circuit spiking
J = np.array([[6.8, -2.5, -2],\
              [-3, 7., -2],\
              [-2.3, -2.5, 4.1]])*1
#J = np.array([[0, -1/3, 1/3],\
#                      [-1/3 ,0 , -1/3],\
#                      [-1/3, 1/3, 0]])
J = np.array([[-.5,-0.9],[0.1,-.5]])
#T, N, dt = 1000, 2, 0.1
#C_ = Circuit_dynamics(T=T, N=N, J=J, A=1, n_pump = 'sine', dt = dt)
#xx,ss,rr,stim = C_.simulation()
T, N, dt = 3, 2, 0.001
C_ = Circuit_dynamics(T=3, N=2, J=J, A=1, n_pump = 'sine', dt = dt)
xx,ss,rr,stim = C_.simulation_pattern('bistable')

# %% GLM inference
pad = 50
nb = 8
GLM_ = GLM_inference(ss, stim, dt, pad, nb)
basis = GLM_.basis_function()
Y, X = GLM_.design_matrix(0, 1, basis)
res = GLM_.MAP_inerence(Y, X, )
Ks, bias = GLM_.recover_kernel(basis,res.x)

# %% analysis
lt =int(T/dt)
ss_recs = np.zeros((N,lt))
#plt.figure()
fig, axs = plt.subplots(N, N+1)
for nn in range(N):
    Y, X = GLM_.design_matrix(nn, 1, basis)
    res = GLM_.MAP_inerence(Y, X, )
    Ks, bias = GLM_.recover_kernel(basis,res.x)
    for jj in range(0,N+1):
        axs[nn, jj].plot(Ks[jj,:])
    
    ss_recs[nn,:],fx = GLM_.Poisson_GLM(res.x, X)
    

# %% goodness of fit
#check that spike train can be recovered!
#Y_rec = GLM_.Poisson_GLM(res.x, X)
plt.figure()
plt.subplot(211)
#plt.imshow(ss,aspect='auto')
plt.plot(ss.T,linewidth=3)
plt.subplot(212)
ss_recs[ss_recs>1] = 1
#plt.imshow(ss_recs,aspect='auto')
plt.plot(ss_recs.T,linewidth=3)


# %%
###############################################################################
# %% with ground truth kernels
T = 1000
dt = 0.1
N = 2
pad = 50
nb = 8
GLM_ = GLM_inference(ss, stim, 0.1, pad, nb)
basis = GLM_.basis_function()
kernels = np.zeros((N,N+1,pad))
for ii in range(N):
    for jj in range(N+1):
        #wks = np.random.randn(nb)*.05
        wks = np.array([-0.1,1,0.5,0.1,0.1,0,0,0])*.1
        kernels[ii,jj,:] = np.dot(wks,basis.T) #- np.abs(np.sum(np.dot(wks,basis.T)))
        if ii+1==jj:  #diagonals
            #wks = np.arange(nb,0,-1)*.1
            wks = np.array([1,-0.9,-0.5,-0.1,0.1,0,0,0])*.5
            kernels[ii,jj,:] = np.dot(wks,basis.T) #- np.abs(np.sum(np.dot(wks,basis.T)))
mus = np.random.randn(N)
#kernels[:,0,-1] = mus
C_ = Circuit_dynamics(T=T, N=2, J=None, A=.1, n_pump = 'noise', dt = dt)
Psks, spks, stim, kernels = C_.coupled_GLM(kernels, mus)

plt.figure()
plt.plot(spks.T)
# %%
GLM_ = GLM_inference(spks, stim, 0.1, pad, nb)
#Y, X = GLM_.design_matrix(0, 1, basis)
#res = GLM_.MAP_inerence(Y, X, )
#Ks, bias = GLM_.recover_kernel(basis,res.x)

# %% analysis
spk_recs = np.zeros((N,int(T/dt)))
#plt.figure()
fig, axs = plt.subplots(N, N+1)
for nn in range(N):
    Y, X = GLM_.design_matrix(nn, 1, basis)
    res = GLM_.MAP_inerence(Y, X, )
    Ks, bias = GLM_.recover_kernel(basis,res.x)
    for jj in range(0,N+1):
        axs[nn, jj].plot(kernels[nn,jj,:])  #true kernel
        axs[nn, jj].plot(Ks[jj,:],'--')  #infererred kernel
    print('inferred: ', bias)
    print('ture baseline:', mus[nn])
    
    spk_recs[nn,:],_ = GLM_.Poisson_GLM(res.x, X)

plt.figure()
plt.subplot(211)
plt.imshow(spks,aspect='auto')
plt.subplot(212)
#ss_recs[spk_recs>1] = 1
plt.imshow(spk_recs,aspect='auto')

# %% encoding & decoding
#vary input end to MAP for decoding


# %% NL dynamical tasks
#stability and others~~
