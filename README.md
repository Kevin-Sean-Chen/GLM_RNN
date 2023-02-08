# GLM_RNN
Exploring the dynamical repertoire of GLM network, properties of stochastic spiking networks, and tasks for RNNs

GLMnet
----
Function and scripts to generate recurrent neural network (RNN) constituted by generalized linear models (GLM); then perform inference on parameters given target spikng patterns or readout time series. Some main scripts I am working on:

- ``GLM_RNN.py`` - Is a simple demonstration for generative model with spikes either from ground-truth parameters or latent time seires. We then perform inference and seek to reconstruct similar spiking patterns.

- ``brunel_GLM.py`` - Another demonstration for different target spiking patterns (AI, AR, SI, SR) described in the Brunel 2000 paper. We then perform inference with GLM network to capture these population spiking patterns.

- ``lowrank_GLM.py`` - Simulates spikes with low-rank connectivity and compares inference performance with or without low-rank regularization.

- ``RSNN.py.py`` - Implementing Bellec et al. 2022 method of pseudo-gradient and back-prop through time to train GLM networks with perceptual decision making tasks.

- ``GFORCE_seq.py`` - Demo for GLM-FORCE algorithm with target that is sequential spiking pattern.

Currently working on packing functions/classes for this:

    from GLM_RNN.GLMnet import nonlinearity

    nonlinearity()
    
FORCE
----
Scripts for FORCE learning algorithm and its generalization to GLM networks, different target dynamics, and network constraints.

fFORCE
----
Preliminary tests for full-FORCE learning algorithm and its generalization to GLM networks. The goal is to jointly learn latent 'hints' for training.

unittest
----
Still working on examples for unit-test and packaging.
