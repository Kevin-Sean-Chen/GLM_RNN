# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 20:10:26 2023

@author: kevin
"""
import autograd.numpy as np
import autograd.numpy.random as npr
import ssm.stats as stats
from ssm.optimizers import adam, bfgs, rmsprop, sgd, lbfgs
from ssm.util import random_rotation, ensure_args_are_lists, logistic, logit, one_hot

# %%
class Observations(object):
    # K = number of discrete states
    # D = number of observed dimensions
    # M = exogenous input dimensions (the inputs modulate the probability of discrete state transitions via a multiclass logistic regression)

    def __init__(self, K, D, M):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    def permute(self, perm):
        pass

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None, init_method="random"):
        Ts = [data.shape[0] for data in datas]

        # Get initial discrete states
        if init_method.lower() == 'kmeans':
            # KMeans clustering
            from sklearn.cluster import KMeans
            km = KMeans(self.K)
            km.fit(np.vstack(datas))
            zs = np.split(km.labels_, np.cumsum(Ts)[:-1])

        elif init_method.lower() =='random':
            # Random assignment
            zs = [npr.choice(self.K, size=T) for T in Ts]

        else:
            raise Exception('Not an accepted initialization type: {}'.format(init_method))

        # Make a one-hot encoding of z and treat it as HMM expectations
        Ezs = [one_hot(z, self.K) for z in zs]
        expectations = [(Ez, None, None) for Ez in Ezs]

        # Set the variances all at once to use the setter
        self.m_step(expectations, datas, inputs, masks, tags)

    def log_prior(self):
        return 0

    def log_likelihoods(self, data, input, mask, tag):
        raise NotImplementedError

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        raise NotImplementedError

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="bfgs", **kwargs):
        """
        If M-step cannot be done in closed form for the observations, default to SGD.
        """
        optimizer = dict(adam=adam, bfgs=bfgs, lbfgs=lbfgs, rmsprop=rmsprop, sgd=sgd)[optimizer]

        # expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, _, _) \
                in zip(datas, inputs, masks, tags, expectations):
                lls = self.log_likelihoods(data, input, mask, tag)
                elbo += np.sum(expected_states * lls)
            return elbo

        # define optimization target
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        self.params = optimizer(_objective, self.params, **kwargs)

    def smooth(self, expectations, data, input, tag):
        raise NotImplementedError

    def neg_hessian_expected_log_dynamics_prob(self, Ez, data, input, mask, tag=None):
        raise NotImplementedError
        
    
# %%    
class GLM_PoissonObservations(Observations):

    def __init__(self, K, D, M):
        super(GLM_PoissonObservations, self).__init__(K, D, M)
        self.Wk = npr.randn(K, D, M)
        self.K, self.D, self.M = K, D, M   #K-state, D-dim output, M-dim input
        self.nl_type = 'exp'

    @property
    def params(self):
        return self.Wk

    @params.setter
    def params(self, value):
        self.Wk = value

    def permute(self, perm):
        self.Wk = self.Wk[perm]
        
#    def calculate_logPoisson(self, input):
#        # np.sum(Y * np.log(f(X@w)) - f(X@w)*dt - sp.special.gammaln(Y+1) + Y*np.log(dt))
#        Wk_trans = np.transpose(self.Wk, (1,0,2))

    def log_likelihoods(self, data, input, mask, tag):
        lambdas = self.nonlinearity((self.Wk @ input.T) * 1.0)  #exponential Poisson nonlinearity
#        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        mask = np.ones_like(data, dtype=bool) if mask is None else mask
        ############################################ fix log-ll here ##################################
        return stats.poisson_logpdf(data[None,:,:], np.transpose(lambdas,(0,2,1)), mask=mask[None,:,:]).T

    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
#        assert self.D == 1, "InputDrivenObservations written for D = 1!"
        if input.ndim == 1 and input.shape == (self.M,): # if input is vector of size self.M (one time point), expand dims to be (1, M)
            input = np.expand_dims(input, axis=0)
        lambdas = self.nonlinearity(self.Wk @ input.T)
        if lambdas[z].shape[1]==1:
            lambz = np.squeeze(lambdas[z])  # dealing with 1-D indexiing
        else:
            lambz = lambdas[z]
        return npr.poisson(lambz)   # y = Poisson(exp(W @ x)) for classic exp-Poisson spiking

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs)
##        x = np.concatenate(datas)
##        weights = np.concatenate([Ez for Ez, _, _ in expectations])
##        for k in range(self.K):
##            self.log_lambdas[k] = np.log(np.average(x, axis=0, weights=weights[:,k]) + 1e-16)
#        
#        X = np.concatenate(inputs)
#        y = np.concatenate(datas)
#        expt = np.concatenate([Ez for Ez, _, _ in expectations]) #expectation=gamma,xi,ll
##        print(X.shape)
##        print(y.shape)
##        print(expt.shape)
#        for k in range(self.K):
#            weigthed_y = y[:,0]*(expt[:,k]) /sum(expt[:,k])  ## weighted here
##            print(weigthed_y.shape)
#            what, bhat = fit_scalar_glm(X, weigthed_y, model="poisson", mean_function="exp")
#            self.Wk[k] = what


    def smooth(self, expectations, data, input, tag):
        """
        Compute the mean observation under the posterior distribution
        of latent discrete states.
        """
#        return expectations.dot(np.exp(self.log_lambdas))
        raise NotImplementedError
        
    def nonlinearity(self,x):
        if self.nl_type == 'exp':
            nl = np.exp(x)
        return nl

# %%
class GLM_GaussianObservations(Observations):
    def __init__(self, K, D, M):
        super(GLM_GaussianObservations, self).__init__(K, D, M)
        self.Wk = npr.randn(K, D, M)
        self.K, self.D, self.M = K, D, M   #K-state, D-dim output, M-dim input
        self.nl_type = 'exp'
    
# %%
class InputVonMisesObservations(Observations):
    def __init__(self, K, D, M):
        super(InputVonMisesObservations, self).__init__(K, D, M)
        self.mus = npr.randn(K, D, M)  #this is a kernel acting on input for vonMises mean
        self.log_kappas = np.log(-1*npr.uniform(low=-1, high=0, size=(K,D)))  ###change this dimension for cov matrix

    @property
    def params(self):
        return self.mus, self.log_kappas  #make mus a vector here for GLM-like emission

    @params.setter
    def params(self, value):
        self.mus, self.log_kappas = value

    def permute(self, perm):
        self.mus = self.mus[perm]
        self.log_kappas = self.log_kappas[perm]

    def log_likelihoods(self, data, input, mask, tag):
        mus, kappas = self.mus, np.exp(self.log_kappas)
        ###
        driven_angle = mus @ input.T  # K x D x T
        mask = np.ones_like(data, dtype=bool) if mask is None else mask    
        sigmas_t = np.repeat(kappas[None, :,:], input.T.shape[-1], axis=0)
        return stats.multivariate_normal_logpdf(data[:,:,None], np.transpose(driven_angle,(2,0,1)), sigmas_t[:,:,:,None])


    def sample_x(self, z, xhist, input=None, tag=None, with_noise=True):
        D, mus, kappas = self.D, self.mus, np.exp(self.log_kappas)
        assert D == 1, "InputDrivenObservations written for D = 1!"
        if input.ndim == 1 and input.shape == (self.M,): # if input is vector of size self.M (one time point), expand dims to be (1, M)
            input = np.expand_dims(input, axis=0)
        ###
        driven_angle = mus @ input.T
        kappas_t = np.repeat(kappas[:,:,None], input.T.shape[-1], axis=2)  #time-independent
        return npr.normal(driven_angle[z], kappas_t[z] )#, D)  #change to npr.vonmesis

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Observations.m_step(self, expectations, datas, inputs, masks, tags, optimizer="bfgs", **kwargs)

#        x = np.concatenate(datas)
#        weights = np.concatenate([Ez for Ez, _, _ in expectations])  # T x D
#        assert x.shape[0] == weights.shape[0]

#        # convert angles to 2D representation and employ closed form solutions
#        x_k = np.stack((np.sin(x), np.cos(x)), axis=1)  # T x 2 x D
#
#        r_k = np.tensordot(weights.T, x_k, axes=1)  # K x 2 x D
#        r_norm = np.sqrt(np.sum(np.power(r_k, 2), axis=1))  # K x D
#
#        mus_k = np.divide(r_k, r_norm[:, None])  # K x 2 x D
#        r_bar = np.divide(r_norm, np.sum(weights, 0)[:, None])  # K x D
#
#        mask = (r_norm.sum(1) == 0)
#        mus_k[mask] = 0
#        r_bar[mask] = 0
#
#        # Approximation
#        kappa0 = r_bar * (self.D + 1 - np.power(r_bar, 2)) / (1 - np.power(r_bar, 2))  # K,D
#
#        kappa0[kappa0 == 0] += 1e-6
#
#        for k in range(self.K):
#            self.mus[k] = np.arctan2(*mus_k[k])  #
#            self.log_kappas[k] = np.log(kappa0[k])  # K, D

        # X = np.concatenate(inputs)
        # y = np.concatenate(datas)
        # expt = np.concatenate([Ez for Ez, _, _ in expectations])
        # for k in range(self.K):
        #     weigthed_y = y[:,0]*(expt[:,k]) /sum(expt[:,k])  ## weighted here
        #     what, bhat = fit_scalar_glm(X, weigthed_y, model="gaussian", mean_function="identity")
        #     self.mus[k] = what

    def smooth(self, expectations, data, input, tag):
        mus = self.mus
        return expectations.dot(mus)