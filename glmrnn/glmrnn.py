#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:47:29 2023

@author: kschen
"""
import numpy as np

class glmrnn:
    
    def __init__(self,N, T, dt, k, kernel_type='tau', nl_type='exp', spk_type="Poisson"):
        
        self.N = N
        self.T = T
        self.dt = dt
        self.kernel, self.nl_type, self. spk_type = kernel_type, nl_type, spk_type
        
    def forward(self):
        
        return
    
    @classmethod
    def nonlinearity(cls):
        
        return
    
    @classmethod
    def spiking(cls):
        
        return
    
    @classmethod
    def kernel(cls):
        
        return
    
    def log_likelihood(self):
        
        return
    