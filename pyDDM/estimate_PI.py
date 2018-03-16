#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:27:33 2018

@author: rishijumani
"""

import numpy as np
import copy
import scipy.stats as st

import pickle

import data_scale
import data_unscale
import assess_dist
import fit_ML_laplace

# add libsvm to path
import sys
sys.path.append('/Volumes/A_2TB/Work/Papers_Code/bin/libsvm-3.22/python')

from svmutil import *



def estimate_PI(x, y, opt, cv_out, x_te):
	
	a = 0.01
	b = 0.99
	y = y.reshape(y.shape[0],1)
	dat = np.hstack((x, y))
	res0 = copy.deepcopy(y)
	dat, dmin, dmax = data_scale.data_scale(dat, a, b)
	x = dat[:,:-1]
	y = dat[:,-1]
	
	y = y.reshape(y.shape[0],1)
	
	drange = dmax - dmin
	
	x_te = (x_te - np.ones((x_te.shape[0], 1))*dmin[:,:-1])*(b - a)/ \
			(np.ones((x_te.shape[0], 1))*drange[:,:-1]) + a*np.ones(x_te.shape)
	
	
	# obtain CV errors
	
	y_te = np.zeros(x.shape[0])
	scores = np.zeros((3, 3))
	
	pi = np.zeros((x_te.shape[0], 2))
	
	if opt['ddm'] == 2:
		
		nfold = opt['nfold']
		cvid = np.random.permutation(np.shape(x)[0])
		cvid = np.mod(cvid, nfold)
		
		for cv in range(nfold):
			
			te_id = cvid == cv
			tr_id = np.logical_not(te_id)
			cv_xtr = x[tr_id, :]
			cv_xte = x[te_id, :]
			cv_ytr = y[tr_id, :]
			cv_yte = y[te_id, :]
			
			cv_ytr = cv_ytr.reshape(cv_ytr.shape[0])
			cv_yte = cv_yte.reshape(cv_yte.shape[0])
			
			cmd = ['-q -s 3 -t 2 -c', str(cv_out['par'][0]), '-g',
					str(cv_out['par'][1][0]) , '-p', str(cv_out['par'][2][0])]
					
			cmd = " ".join(cmd)
			
			l_cv_ytr = cv_ytr.tolist()
			l_cv_xtr = cv_xtr.tolist()
			l_cv_yte = cv_yte.tolist()
			l_cv_xte = cv_xte.tolist()

			model = svm_train(l_cv_ytr, l_cv_xtr, cmd)
			y_te[te_id], accu2, prob = svm_predict(l_cv_yte, l_cv_xte, model)
			
		
		
		y_te = data_unscale.data_unscale(y_te, a, b, x.shape[1], cv_out)
		y_te = y_te.reshape(y_te.shape[0],1)
		
		res = res0 - y_te
		
		with open('dist.pkl', 'w') as f:  # Python 3: open(..., 'wb')
			pickle.dump([res], f)
		
		# estimate distribution
		
		W, A, LL = assess_dist.assess_dist(res)
		
		scores = [W, A, LL]
		
		dist_id = np.where(LL == np.min(LL))
		
		s = (1 - opt['cl'])/2
		
		if dist_id[0][0] == 0:
			# Laplace
			lap = fit_ML_laplace.fit_ML_laplace(res)
			lmu = lap['u']
			lsig = lap['b']
			
			pi[:,0] = lmu + lsig*np.log(2*s)
			pi[:,1] = lmu - lsig*np.log(2*s)
			
		elif dist_id[0][0] == 1:
			
			phat = st.norm.fit(res)
			mle = st.norm.nnlf(phat, res)  # computes MLE
			
			mu = phat[0]
			sig = phat[1]
			
			pi[:, 0] = st.norm.ppf(s, mu, sig)
			pi[:, 1] = st.norm.ppf(1-s, mu, sig)		
			
		elif dist_id[0][0] == 2:
			pars = st.cauchy.fit(res)
			x0 = pars[0]
			gamma = pars[1]
			pi[:,0] = st.cauchy.ppf(s, x0, gamma)
			pi[:,1] = st.cauchy.ppf(1-s, x0, gamma)
			
			
	return scores, pi
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
	
	
	
	
	