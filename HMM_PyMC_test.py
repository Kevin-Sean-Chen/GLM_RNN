# -*- coding: utf-8 -*-
"""
Created on Sat May 29 09:26:33 2021

@author: kevin
"""

import numpy as np
import scipy as sp
import pymc3 as pm

from typing import List
from scipy.stats import norm, multinomial

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rc('xtick', labelsize=35) 
matplotlib.rc('ytick', labelsize=35) 

import matplotlib as mpl
mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True

%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = 'retina'

# %% functions for HMM generative model
def equilibrium_distribution(p_transition):
    """This implementation comes from Colin Carroll, who kindly reviewed the notebook"""
    n_states = p_transition.shape[0]
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )
    # Moore-Penrose pseudoinverse = (A^TA)^{-1}A^T
    pinv = np.linalg.pinv(A)
    # Return last row
    return pinv.T[-1]

def markov_sequence(p_init: np.array, p_transition: np.array, sequence_length: int) -> List[int]:
    """
    Generate a Markov sequence based on p_init and p_transition.
    """
    if p_init is None:
        p_init = equilibrium_distribution(p_transition)
    initial_state = list(multinomial.rvs(1, p_init)).index(1)

    states = [initial_state]
    for _ in range(sequence_length - 1):
        p_tr = p_transition[states[-1]]
        new_state = list(multinomial.rvs(1, p_tr)).index(1)
        states.append(new_state)
    return states

#def poisson_emissions(states: List[int], lam: List[float]) -> List[int]:
#    emissions = []
#    for state in states:
#        rate = lam[state]
#        e = poisson.rvs(rate)
#        emissions.append(e)
#    return emissions

def gaussian_emissions(states: List[int], mus: List[float], sigmas: List[float]) -> List[float]:
    emissions = []
    for state in states:
        loc = mus[state]
        scale = sigmas[state]
        e = norm.rvs(loc=loc, scale=scale)
        emissions.append(e)
    return emissions

# %% for inference
def state_logp(states, p_transition):
    logp = 0

    # states are 0, 1, 2, but we model them as [1, 0, 0], [0, 1, 0], [0, 0, 1]
    states_oh = np.eye(len(p_transition))
    for curr_state, next_state in zip(states[:-1], states[1:]):
        p_tr = p_transition[curr_state]
        logp += multinomial(n=1, p=p_tr).logpmf(states_oh[next_state])
    return logp

def state_logp_vect(states, p_transition):
    states_oh = np.eye(len(p_transition))
    p_tr = p_transition[states[:-1]]
    obs = states_oh[states[1:]]
    return np.sum(multinomial(n=1, p=p_tr).logpmf(obs))

def initial_logp(states, p_transition):
    initial_state = states[0]
    states_oh = np.eye(len(p_transition))
    eq_p = equilibrium_distribution(p_transition)
    return (
        multinomial(n=1, p=eq_p)
        .logpmf(states_oh[initial_state].squeeze())
    )
    
def markov_state_logp(states, p_transition):
    return (
        state_logp_vect(states, p_transition)
        + initial_logp(states, p_transition)
    )
    
def gaussian_logp_vect(states, mus, sigmas, emissions):
    mu = mus[states]
    sigma = sigmas[states]
    return np.sum(norm(mu, sigma).logpdf(emissions))

def gaussian_emission_hmm_logp(states, p_transition, mus, sigmas, emissions):
    return markov_state_logp(states, p_transition) + gaussian_logp_vect(states, mus, sigmas, emissions)

def solve_equilibrium(n_states, p_transition):
    A = tt.dmatrix('A')
    A = tt.eye(n_states) - p_transition + tt.ones(shape=(n_states, n_states))
    p_equilibrium = pm.Deterministic("p_equilibrium", sla.solve(A.T, tt.ones(shape=(n_states))))
    return p_equilibrium


# %% PyMC3
import theano.tensor as tt
import theano.tensor.slinalg as sla  # theano-wrapped scipy linear algebra
import theano.tensor.nlinalg as nla  # theano-wrapped numpy linear algebra
import theano

# %%

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

class HMMStates(pm.Categorical):
    def __init__(self, p_transition, p_equilibrium, n_states, *args, **kwargs):
        """You can ignore this section for the time being."""
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_transition = p_transition
        self.p_equilibrium = p_equilibrium
        # This is needed
        self.k = n_states
        # This is only needed because discrete distributions must define a mode.
        self.mode = tt.cast(0,dtype='int64')

    def logp(self, x):
        """Focus your attention here!"""
        p_eq = self.p_equilibrium
        # Broadcast out the transition probabilities,
        # so that we can broadcast the calculation
        # of log-likelihoods
        p_tr = self.p_transition[x[:-1]]

        # the logp of the initial state evaluated against the equilibrium probabilities
        initial_state_logp = pm.Categorical.dist(p_eq).logp(x[0])

        # the logp of the rest of the states.
        x_i = x[1:]
        ou_like = pm.Categorical.dist(p_tr).logp(x_i)
        transition_logp = tt.sum(ou_like)
        return initial_state_logp + transition_logp
    
class HMMGaussianEmissions(pm.Continuous):
    def __init__(self, states, mu, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = states
        # self.rate = rate
        self.mu = mu
        self.sigma = sigma

    def logp(self, x):
        """
        x: observations
        """
        states = self.states
        # rate = self.rate[states]  # broadcast the rate across the states.
        mu = self.mu[states]
        sigma = self.sigma[states]
        return tt.sum(pm.Normal.dist(mu=mu, sigma=sigma).logp(x))

# %%
###############################################################################
# %% DATA
p_init = np.array([0.1, 0.8, 0.1])
p_transition = np.array(
    [[0.90, 0.05, 0.05],
     [0.01, 0.90, 0.09],
     [0.07, 0.03, 0.9]]
)
states = markov_sequence(p_init, p_transition, sequence_length=1000)
gaussian_ems = gaussian_emissions(states, mus=[1, 0, -1], sigmas=[0.2, 0.5, 0.1])

def plot_emissions(states, emissions):
    fig, axes = plt.subplots(figsize=(16, 8), nrows=2, ncols=1, sharex=True)

    axes[0].plot(states)
    axes[0].set_title("States")
    axes[1].plot(emissions)
    axes[1].set_title("Emissions")
    sns.despine();

plot_emissions(states, gaussian_ems)

# %% INFERENCE
n_states = 3
with pm.Model() as model:
    # Priors for transition matrix
    p_transition = pm.Dirichlet("p_transition", a=tt.ones((n_states, n_states)), shape=(n_states, n_states))

    # Solve for the equilibrium state
    p_equilibrium = solve_equilibrium(n_states, p_transition)

    # HMM state
    hmm_states = HMMStates(
        "hmm_states",
        p_transition=p_transition,
        p_equilibrium=p_equilibrium,
        n_states=n_states,
        shape=(len(gaussian_ems),)
    )

    # Prior for mu and sigma
    mu = pm.Normal("mu", mu=0, sigma=1, shape=(n_states,))
    sigma = pm.Exponential("sigma", lam=2, shape=(n_states,))

    # Observed emission likelihood
    obs = HMMGaussianEmissions(
        "emission",
        states=hmm_states,
        mu=mu,
        sigma=sigma,
        observed=gaussian_ems
    )
    
# %%
with model:
    trace = pm.sample(2000, tune=1000, cores=1)
# %%
import arviz as az
az.plot_trace(trace, var_names=["mu"]);

az.plot_forest(trace, var_names=["sigma"]);

# %% play with other distributions!!!
