# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:51:22 2020

@author: kevin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from dotmap import DotMap

import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")


from pyglmnet import GLM, simulate_glm
from pyglmnet import GLMCV
from pyglmnet import GLM

global eps
eps = 10**-15  #flow-point precisio

# %% Poisson neural network with TM synapse
N = 2
dt = 0.1  #ms
T = 1000
time = np.arange(0,T,dt)
lt = len(time)

v = np.zeros((N,lt))  #voltage
spk = np.zeros_like(v)  #spikes
x = np.zeros_like(v)  #synaptic resource
u = np.zeros_like(v)  #synaptic efficacy
x[:,0] = np.random.rand(N)
u[:,0] = np.random.rand(N)
rate = np.zeros_like(v)  #spike rate
v[:,0] = np.random.randn(N)*1
#J = np.random.randn(N,N)
J = np.array([[3.1, -1.5],\
              [-1.5, 3.1]])
#J = np.array([[1, -0],\
#          [-0, 1.]])
J = J.T*5
noise = 0.
stim = np.random.randn(lt)*30  #np.random.randn(N,lt)*20.
taum = 2  #5 ms
tauD = 500
tauF = 100
U = 0.5
E = 1

eps = 10**-15
def LN(x):
    """
    nonlinearity
    """
    ln = 1/(1+np.exp(-x*1.+eps))   #logsitic
#    ln = np.log(1+np.exp(x)+eps)
#    ln = np.array([max(min(100,xx),0) for xx in x])  #ReLu
    return np.random.poisson(ln) #ln  #Poinsson emission

def spiking(ll,dt):
    """
    Given Poisson rate (spk per second) and time steps dt return binary process with the probability
    """
    N = len(ll)
    spike = np.random.rand(N) < ll*dt  #for Bernouli process
    return spike

###iterations for neural dynamics
for tt in range(0,lt-1):
    v[:,tt+1] = v[:,tt] + dt/taum*( -v[:,tt] + (np.matmul(J,LN(x[:,tt]*u[:,tt]*v[:,tt]))) + stim[tt]*np.array([1,1]) + noise*np.random.randn(N)*np.sqrt(dt))
    spk[:,tt+1] = spiking(LN(x[:,tt+1]),dt)
    rate[:,tt+1] = LN(x[:,tt+1])
    u[:,tt+1] = u[:,tt] + dt*( (U-u[:,tt])/tauF + (1-u[:,tt])*U*spk[:,tt] )
    x[:,tt+1] = x[:,tt] + dt*( (1-x[:,tt])/tauD - u[:,tt]*x[:,tt]*spk[:,tt] )
    
plt.figure()
plt.subplot(411)
plt.imshow(spk,aspect='auto');
plt.subplot(412)
plt.imshow(rate,aspect='auto');
plt.subplot(413)
plt.plot(time,x.T);
plt.xlim([0,time[-1]])
plt.subplot(414)
plt.plot(time,stim.T);
plt.xlim([0,time[-1]])

# %% TM-GLM inference
def log_prior_theta(gamma1,gamma2,beta1,beta2):
    D, F = np.random.gamma(gamma1, gamma1)
    U, f = np.ranodm.beta(1.01, 1.01, 2)
    theta = D,F,U,f
    if D>2 or F>2:
        log_prior = 0  #for -inf log probability
    else:
        log_prior = sp.stats.gamma.logpdf(D, 1.2, 0, 2) + sp.stats.gamma.logpdf(F, 1.2, 0, 2) \
                    + sp.stats.beta.logpdf(U, 1.01, 0, 1.) + sp.stats.beta.logpdf(f, 1.01, 0, 1.)
    return log_prior, theta

def neg_ll_GLM(THETA, y, X, theta_):
    D,F,U,f = theta_
    k = kernel(THETA)
    v = LN(np.matmul(X,k))  #nonlinear function
    nl_each = -(np.matmul(y.T, np.log(v+eps)) - np.sum(v))  #Poisson negative log-likelihood
    #nl_each = -( np.matmul(Y.T, np.log(v+eps)) - np.matmul( (1-Y).T, np.log(1-v+eps)) )  #Bernouli process of binary spikes
    nl = nl_each.sum()
    return nl

def kernel(THETA):
    return k

def eval_TM(D,F,U,f):
    return

def eval_GLM(mu,r,s):
    return

# %%
Max_samp = 1000  #maximum for sample iterations
ii = 0
accepted = []
rejected = []   

while ii < Max_samp:
    ii = ii+1
    ###sampling for theta parameters
    log_prior, theta = log_prior_theta(1.2, 2, 1.01, 1.01)
    theta0 = np.random.randn(phis)
    res = sp.optimize.minimize( neg_ll_GLM, theta0, args=(y, X, theta_))
                            #, method='Nelder-Mead',options={'disp':True,'maxiter':3000})#
                            #method="BFGS") #, tol=1e-3, options={'disp':True,'gtol':1e-2})#
    log_like = -neg_ll_GLM(res.x, y, X, theta_)
    log_post = log_prior + log_like
    
    log_prior_new, theta_new =  transition_model(theta)    
    log_lik_new = likelihood_computer(theta_new, data) 
    log_post_new = log_prior_new + log_lik_new
    
    if (acceptance_rule(log_post, log_post_new)):            
        theta = theta_new
        accepted.append(theta_new)
    else:
        rejected.append(theta_new)
            
            

# %%
#The tranistion model defines how to move from sigma_current to sigma_new
transition_model = lambda x: [x[0],np.random.normal(x[1],0.5,(1,))]  #Gaussian iid from the previous point

def prior(x):
    #x[0] = mu, x[1]=sigma (new or current)
    #returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    #returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    #It makes the new sigma infinitely unlikely.
    if(x[1] <=0):
        return 0
    return 1

#Computes the likelihood of the data given a sigma (new or current) according to equation (2)
def manual_log_like_normal(x,data):
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    return np.sum(-np.log(x[1] * np.sqrt(2* np.pi) )-((data-x[0])**2) / (2*x[1]**2))

#Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow.
def log_lik_normal(x,data):
    #x[0]=mu, x[1]=sigma (new or current)
    #data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0],x[1]).pdf(data)))


#Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new>x:
        return True
    else:
        accept=np.random.uniform(0,1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return (accept < (np.exp(x_new-x)))


def metropolis_hastings(likelihood_computer,prior, transition_model, param_init,iterations,data,acceptance_rule):
    # likelihood_computer(x,data): returns the likelihood that these parameters generated the data
    # transition_model(x): a function that draws a sample from a symmetric distribution and returns it
    # param_init: a starting sample
    # iterations: number of accepted to generated
    # data: the data that we wish to model
    # acceptance_rule(x,x_new): decides whether to accept or reject the new sample
    x = param_init
    accepted = []
    rejected = []   
    for i in range(iterations):
        x_new =  transition_model(x)    
        x_lik = likelihood_computer(x,data)
        x_new_lik = likelihood_computer(x_new,data) 
        if (acceptance_rule(x_lik + np.log(prior(x)),x_new_lik+np.log(prior(x_new)))):            
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)            
                
    return np.array(accepted), np.array(rejected)
# %%
mod1=lambda t:np.random.normal(10,3,t)

#Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(30000)
#Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 30000, 1000)]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.hist( observation,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Distribution of 1000 observations sampled from a population of 30,000 with mu=10, sigma=3")
mu_obs=observation.mean()
mu_obs

accepted, rejected = metropolis_hastings(manual_log_like_normal,prior,transition_model,[mu_obs,0.1], 50000,observation,acceptance)
