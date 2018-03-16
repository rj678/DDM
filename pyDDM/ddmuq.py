#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:00:32 2018

@author: rishijumani
"""

# Usage:
# 
# training: [ybar pi perf] = ddmuq(x,yobs,ysim,opt)
# testing: [ybar_te pi_te perf_te] = ddmuq(x,yobs,ysim,opt,x_te,ysim_te,yobs_te)
# forecasting: [ybar_te pi_te] = ddmuq(x,yobs,ysim,opt,x_te,ysim_te) 

import numpy as np


import performance_assess
import ddm_train
import ddm_test
import estimate_PI


def ddmuq(x=None,yobs=None,ysim=None,opt=None,x_te=None,ysim_te=None,yobs_te=None):
	

	
	# pre-processing
	print('ReResidual analysis for training period:')
	y = performance_assess.performance_assess(yobs,ysim)	
	print 'PBIAS = {:.6f}  MAE = {:.6f}    NSE = {:.6f}'.format(y[0], y[1], y[2])
	
	
	
	# Train the DDM
	res = yobs - ysim
	cv_out = ddm_train.ddm_train(x, res, opt)
	
	
	# predict
	r = ddm_test.ddm_test(x, yobs-ysim, opt, cv_out, x)
	ysim = ysim.reshape(ysim.shape[0],1)
	ybar = ysim + r
	
	yobs = yobs.reshape(yobs.shape[0],1)
	
	if x_te is not None:
		r_te = ddm_test.ddm_test(x, yobs-ysim, opt, cv_out, x_te)
		ysim_te = ysim_te.reshape(ysim_te.shape[0],1)
		ybar_te = ysim_te + r_te
	
	
	# predict

	scores, pi = estimate_PI.estimate_PI(x, yobs - ysim, opt, cv_out, x)
	
	ybar = ybar.reshape(ybar.shape[0])
	
	
	pi[:,0] = ybar + pi[:,0]
	pi[:,1] = ybar + pi[:,1]
	
	ybar_te = ybar_te.reshape(ybar_te.shape[0])
	
	if x_te is not None:
		scores, pi_te = estimate_PI.estimate_PI(x, yobs - ysim, opt, cv_out, x_te)
		pi_te[:,0] = ybar_te + pi_te[:,0]
		pi_te[:,1] = ybar_te + pi_te[:,1]
		
	print('Distribution goodness-of-fit scores:')
	print(scores)
	
	if yobs_te is None:
		perf1 = performance_assess.performance_assess(yobs, ysim)
		perf2 = performance_assess.performance_assess(yobs, ybar)
		perf = [perf1, perf2]
		
	else:
		perf1 = performance_assess.performance_assess(yobs_te, ysim_te)
		perf2 = performance_assess.performance_assess(yobs_te, ybar_te)
		perf = [perf1, perf2]


	
	
	
	return pi




	

	