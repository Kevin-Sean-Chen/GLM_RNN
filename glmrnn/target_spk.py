# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 22:44:33 2023

@author: kevin
"""
import numpy as np
from glmrnn.glmrnn import glmrnn as gr

class target_spk:
    
    def __init__(self, N, T, d, latent_type):
        self.N = N
        self.T = T
        self.d = d
        self.latent_type = latent_type
        self.latent = np.zeros((d,T))
        
    def forward(self, latent):
        """
        forward generation of spiking patterns given latent dynamics
        """
        M = np.random.randn(self.N, self.d)*1.
        b = 0
        # simulate spikes    
        spk = np.zeros((self.N,self.T))  # spike train
        for tt in range(self.T-1):
             spk[:,tt] = gr.spiking(gr.nonlinearity(M*latent[tt]-b))  # latent-driven spikes
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
        
        return latent
    
    def line_attractor(self):
        return
    
    def oscillation(self):
        return
        
    def sequence(self):
        return
    
    def chaotic(self):
        return
    
    def brunel_pattern(self):
        return
    
    def stochastic_states(self):
        # GLM-HMM
        return