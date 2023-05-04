# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:44:33 2023

@author: kevin
"""
import numpy as np
# from glmrnn.glmrnn import glmrnn as gr

class target_spk(object):
    
    def __init__(self, N, T, d, gr):
        self.N = N
        self.T = T
        self.d = d
        # self.latent_type = latent_type
        # self.latent = np.zeros((d,T))
        # self.M = np.random.randn(self.N, self.d)  # loading matrix
        self.my_network = gr  # containing the class for glmrnn settings
        self.M = np.random.randn(N, d)*1.
        
    def _forward(self, latent):
        """
        forward generation of spiking patterns given latent dynamics
        """
#        M = np.random.randn(self.N, self.d)*1.
#        if self.d>1:     
        if self.d==1:
            latent = latent[None,:]
        b = 0
        # simulate spikes    
        spk = np.zeros((self.N,self.T))  # spike train
        # print(latent.shape)
        for tt in range(self.T-1): #test
            spk[:,tt] = self.my_network.spiking(self.my_network.nonlinearity( \
                       (self.M @ latent[:,tt]).squeeze()-b))  # latent-driven spikes
        return spk
        
    def bistable(self):
        """
        Create bi-stable latent from a noise double-well particle
        """
        c = 0.  # posision
        sig = 1.  # noise
        tau_l = 2  # time scale
        dt = 0.1  # time step
        def vf(x):
            """
            Derivitive of focring in a double-well
            """
            return -(x**3 - 2*x - c)
        latent = np.zeros(self.T)
        for tt in range(self.T-1):
            ### simpe SDE
            latent[tt+1] = latent[tt] + dt/tau_l*(vf(latent[tt])) + np.sqrt(dt*sig)*np.random.randn()
        
        spk = self._forward(latent)
        return spk, latent
    
    def line_attractor(self,neur_id):
        """
        bump attractor on a ring with known connectivity (Fiete's method)
        """
        W = self._bump_matrix()
        b = 0.1
        U = np.zeros(self.N)
        U[neur_id] = 1
        spk = np.ones((self.N, self.T))
        rt = np.ones((self.N, self.T))
        ipt = np.random.randn(self.T)*1  # just noise input for now
        ipt[:100] = 2
        for tt in range(self.T-1):
            lamb = W @ rt[:,tt] + b + U*ipt[tt]
            spk[:,tt+1] = self.my_network.spiking(self.my_network.nonlinearity(lamb*self.my_network.dt/self.my_network.dt))
            rt[:,tt+1] = self.my_network.kernel(rt[:,tt] , spk[:,tt]) + np.random.randn(self.N)*0.25
        return spk, ipt
    
    def _bump_matrix(self):
        """
        creating bump/ring attractor connectivity
        """
        thetas = 2*np.pi*np.arange(0,self.N)/self.N
        def bump(theta):
            A = 5
            k1 = 1*1
            k2 = 0.3*1
            return A*np.exp(k1*(np.cos(theta)-1)) - A*np.exp(k2*(np.cos(theta)-1))  # Maxican hat formula
        wij = np.zeros((self.N, self.N))
        for ii in range(self.N):
            for jj in range(self.N):
                wij[ii,jj] = bump(thetas[ii] - thetas[jj])  # connectivity matrix
        return wij
    
    def oscillation(self, period=50):
        """
        sine waves given a period
        """
        time = np.arange(0,self.T)
        latent = 2*np.sin(time/period)
        spk = self._forward(latent)
        return spk, latent
        
    def sequence(self, bsize):
        """
        spread-diagonal pattern
        tiling the time coarse with neurons given bump size
        """
        locs = np.linspace(0, self.T, self.N)
        rt = np.zeros((self.N, self.T))
        timev = np.arange(0,self.T)
        for nn in range(0,self.N):
            rt[nn,:] = np.exp(-(timev-locs[nn])**2 / (2*bsize**2))*5 + np.random.randn(self.T)*0.1
        spk = self.my_network.spiking(self.my_network.nonlinearity(rt))
        return spk, rt
    
    def chaotic(self):
        """
        Lorentz attractor
        """
        dt = 0.01
        xyzs = np.empty((self.T-1 + 1, 3))  # Need one more for the initial values
        xyzs[0] = (0., 1., 1.05)  # Set initial values
        for i in range(self.T-1):
            xyzs[i+1] = xyzs[i] + self._lorenz(xyzs[i])*dt
        latent = xyzs[:,:self.d]#.T.squeeze()  # take 1-3 dimension as the latent
        spk = self._forward(latent)
        return spk, latent
    
    def _lorenz(self, xyz, *, s=10, r=28, b=2.667):
        """
        differential of xyz variables in Lorenz attractor
        """
        x, y, z = xyz
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])
    
    def stochastic_rate(self, ipt, ms=None, eps=0):
        """
        ipt: [0,1]
        Step input in the later half that flips firing rate probablisticlity
        """
        latent = np.ones(self.T)*0.1
        prob = np.random.rand()
        if ms is None:
            if prob > ipt:
                latent[int(self.T/2):] = 1
            else:
                latent[int(self.T/2):] = -1
        else:
            m1,m2 = ms  # if the loading patterns are given
            latent[int(self.T/2):] = 1
            if prob > ipt:
                self.M = m1[:,None]
            else:
                self.M = m2[:,None]
            ####
            # for disengaged state... add another small probablity for not even reacting!
            ####
        if np.random.rand() < eps:
            latent = 0*latent
        spk = self._forward(latent)
        return spk, latent
    
    def brunel_spk(self, phase, lk):
        """
        Latent-driven Poisson spiking patterns to mimic Bruenl 2000 firing patterns,
        with phases SR, AI, SIf, and SIs
        """
        # setup latent, kernels, and inputs
        time = np.arange(0,self.T)*self.my_network.dt
        if phase=='SR':
            latent = 3*np.ones(self.T)
            k_self_spk = -20*np.exp(-np.arange(lk)/20)
            C = np.ones(self.N)
        elif phase=='AI':
            freq = .1
            latent = .2*np.sin(time*freq*(2*np.pi))
            k_self_spk = -1*np.exp(-np.arange(lk)/1)
            C = np.random.randn(self.N)
        elif phase=='SIf':        
            freq = .7
            latent = 2.*np.sin(time*freq*(2*np.pi))
            k_self_spk = -1*np.exp(-np.arange(lk)/1)
            C = np.ones(self.N)
        elif phase=='SIs': 
            freq = .1
            latent = 2*np.sin(time*freq*(2*np.pi))
            k_self_spk = -1*np.exp(-np.arange(lk)/1)
            C = np.ones(self.N)
        
        # simulate spikes
        k_self_spk = np.fliplr(k_self_spk[None,:])[0]
        rt = np.zeros((self.N,self.T))
        spk = np.zeros((self.N,self.T))
        for tt in range(lk,self.T):
            rt[:,tt] = C*latent[tt] + spk[:,tt-lk:tt] @ k_self_spk
            spk[:,tt] = self.my_network.spiking(self.my_network.nonlinearity(rt[:,tt]))
        
        return spk, rt
    
    def stochastic_states(self):
        # GLM-HMM class
        return