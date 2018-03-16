#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:08:41 2018

@author: rishijumani
"""

import copy
import numpy as np
import scipy.stats as st

import fit_ML_laplace



#import pickle
#
#with open('dist.pkl') as f:  # Python 3: open(..., 'rb')
#    res = pickle.load(f)

def assess_dist(res):
	
	# 1. Cramer-von Mises criterion
	
	W1 = 0
	W2 = copy.deepcopy(W1)
	W3 = copy.deepcopy(W1)
	
	res = np.asarray(res)
	
	vres = copy.deepcopy(res)
	
	N = vres.shape[0]
	
	res = np.sort(vres)
	id1 = np.argsort(vres)
	
	
	
	# MLE of parameters of Laplace
	
		# using scipy
#	pars = st.laplace.fit(res)
#	mle = st.laplace.nnlf(pars, res)
#	mle.append(mle)
	
	lap = fit_ML_laplace.fit_ML_laplace(res)
	lmu = lap['u']
	lsig = lap['b']
	
	id1 = res < lmu
	u1 = copy.deepcopy(res)
	u1[id1] = 0.5*np.exp((res[id1] - lmu*np.ones(res[id1].shape[0] ))/lsig)
	u1[np.logical_not(id1)] = 1 - 0.5*np.exp((lmu*
								np.ones(res[np.logical_not(id1)].shape[0]) - 
								res[np.logical_not(id1)])/lsig)
	
	# compute the CVM statistics of Laplace
	
	W1 = np.sum((u1 - (2*(np.arange(N)+1).T - np.ones((N,1)))/2/N)**2)+1/12/N
	
	# MLE of parameters of Gaussian
	
	# using scipy
	phat = st.norm.fit(res)
	mle = st.norm.nnlf(phat, res)  # computes MLE
	#mle.append(mle)
	
	mu = phat[0]
	sig = phat[1]
	
	u2 = st.norm.cdf(res, mu, sig)
	
	# compute the CVM statistics of Gaussian	
	W2 = np.sum((u2 - (2*(np.arange(N)+1).T - np.ones((N, 1)))/2/N)**2)+1/12/N
	
	
	# MLE parameters of Cauchy	
	pars = st.cauchy.fit(res)
	x0 = pars[0]
	gamma = pars[1]
	
	u3 = 1/np.pi*np.arctan((res - x0*np.ones((N,1)))/gamma) + 0.5*np.ones((N,1))
	
	# compute the CVM statistics of Cauchy	
	W3 = np.sum((u3 - (2*(np.arange(N) + 1).T - np.ones((N,1)))/2/N)**2) + 1/12/N
	
	W = [W1, W2, W3]
	
	
	
	# 2. Anderson-Darling
	
	A = np.zeros(3)	
	N = vres.shape[0]	
	res = np.sort(vres)	
	id1 = np.argsort(vres)
	
	# MLE of parameters of Laplace
	lap = fit_ML_laplace.fit_ML_laplace(res)
	lmu = lap['u']
	lsig = lap['b']
	
	id1 = res < lmu
	u1 = copy.deepcopy(res)
	u1[id1] = 0.5*np.exp((res[id1] - lmu*np.ones(res[id1].shape[0] ))/lsig)
	u1[np.logical_not(id1)] = 1 - 0.5*np.exp((lmu*
								np.ones(res[np.logical_not(id1)].shape[0]) - 
								res[np.logical_not(id1)])/lsig)
	
	u1i = u1[::-1]
	
	# compute the CVM statistics of Laplace
	A[0] = -N - 1/N*np.sum((np.arange(1, 2*N - 1, 2)).T*(np.log(u1) + np.log(1 - u1i)))

	
	# MLE of parameters of Gaussian
	phat = st.norm.fit(res)
	mle = st.norm.nnlf(phat, res)  # computes MLE
	
	mu = phat[0]
	sig = phat[1]
	
	u2 = st.norm.cdf(res, mu, sig)
	u2i = u2[::-1]
	
	# compute the CVM statistics of Gaussian
	A[1] = -N - 1/N*np.sum((np.arange(1, 2*N - 1, 2)).T*(np.log(u2) + np.log(1 - u2i)))
	
	# MLE of parameters of Cauchy
	pars = st.cauchy.fit(res)
	x0 = pars[0]
	gamma = pars[1]
	
	u3 = 1/np.pi*np.arctan((res - x0*np.ones((N,1)))/gamma) + 0.5*np.ones((N,1))
	u3i = u3[::-1]
	
	# compute the CVM statistics of Cauchy
	A[1] = -N - 1/N*np.sum((np.arange(1, 2*N - 1, 2)).T*(np.log(u3) + np.log(1 - u3i)))
	
	
	
	# Likelihood
	LL = np.zeros(3)
	
	N = vres.shape[0]
	
	res = np.sort(vres)	
	id1 = np.argsort(vres)
	
	# MLE of parameters of Laplace
	lap = fit_ML_laplace.fit_ML_laplace(res)
	lmu = lap['u']
	lsig = lap['b']
	
	# MLE of parameters of Gaussian
	
	phat = st.norm.fit(res)
	mle = st.norm.nnlf(phat, res)  # computes MLE
	
	mu = phat[0]
	sig = phat[1]
	
	# MLE of parameters of Cauchy
	pars = st.cauchy.fit(res)
	x0 = pars[0]
	gamma = pars[1]
	
	LL[0] = np.sum(np.log(np.exp(-np.abs(res - lmu*np.ones(res.shape[0]))/lsig)
						/2/lsig))
	
	LL[1] = np.sum(np.log(st.norm.pdf(res, mu, sig)))
	
	LL[2] = -np.sum(np.log(np.pi*gamma*(np.ones(res.shape[0]) + 
			   (res - x0*np.ones(res.shape[0]))**2/gamma/gamma)))
	
	return W, A, LL
	
	
	
	
	
	
	
	
	
	
	
	
	