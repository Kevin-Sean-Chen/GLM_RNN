# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:44:33 2023

@author: kevin
"""
import numpy as np
from glmrnn.glmrnn import glmrnn as gr

class target_spk(object):
    
    def __init__(self, N, T, d, latent_type, gr):
        self.N = N
        self.T = T
        self.d = d
        self.latent_type = latent_type
        self.latent = np.zeros((d,T))
        self.M = np.random.randn(self.N, self.d)  # loading matrix
        self.my_network = gr  # containing the class for glmrnn settings
        
    def _forward(self, latent):
        """
        forward generation of spiking patterns given latent dynamics
        """
        if self.d>1:
            M = np.random.randn(self.N, self.d)*1.
        elif self.d==1:
            M = np.random.randn(self.N)*1.
        b = 0
        # simulate spikes    
        spk = np.zeros((self.N,self.T))  # spike train
        for tt in range(self.T-1):
             spk[:,tt] = self.my_network.spiking(self.my_network.nonlinearity(M*latent[tt]-b))  # latent-driven spikes
        return spk
        
    def bistable(self):
        """
        Create bi-stable latent from a noise double-well particle
        """
        c = 0.  # posision
        sig = .5  # noise
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
            rt[:,tt+1] = self.my_network.kernel(rt[:,tt] , spk[:,tt])
        return spk, ipt
    
    def _bump_matrix(self):
        """
        creating bump/ring attractor connectivity
        """
        thetas = 2*np.pi*np.arange(0,self.N)/self.N
        def bump(theta):
            A = 1
            k1 = 1
            k2 = 0.3
            return A*np.exp(k1*(np.cos(theta)-1)) - A*np.exp(k2*(np.cos(theta)-1))  # Maxican hat formula
        wij = np.zeros((self.N, self.N))
        for ii in range(self.N):
            for jj in range(self.N):
                wij[ii,jj] = bump(thetas[ii] - thetas[jj])  # connectivity matrix
        return wij
    
    def oscillation(self, period):
        """
        sine waves given a period
        """
        time = np.arange(0,self.T)
        latent = np.sin(time/period)
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
            rt[nn,:] = np.exp(-(timev-locs[nn])**2 / (2*bsize**2))
        spk = self.my_network.spiking(rt)
        return spk, rt
    
    def chaotic(self):
        """
        Lorentz attractor
        """
        return
    
    def brunel_pattern(self):
        """
        SR, SI, AR, AI
        """
        return
    
    def stochastic_states(self):
        # GLM-HMM class
        return