#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:38:44 2018

@author: rishijumani
"""

import numpy as np

import data_scale
import data_unscale

# add libsvm to path
import sys
sys.path.append('/Volumes/A_2TB/Work/Papers_Code/bin/libsvm-3.22/python')

from svmutil import *


def ddm_test(x, y, opt, cv_out, x_te):
	
	a = 0.01
	b = 0.99
	
	y = y.reshape(y.shape[0],1)
	dat = np.hstack((x, y))
	
	dat, dmin, dmax = data_scale.data_scale(dat, a, b)
	
	x = dat[:,:-1]
	y = dat[:,-1]
	
	drange = dmax - dmin
	
	x_te = (x_te - np.ones((x_te.shape[0], 1))*dmin[:,:-1])*(b - a)/ \
			(np.ones((x_te.shape[0], 1))*drange[:,:-1]) + a*np.ones(x_te.shape)
			
	if opt['ddm'] == 2:
		cmd = ['-q -s 3 -t 2 -c', str(cv_out['par'][0]), '-g',
					str(cv_out['par'][1][0]) , '-p', str(cv_out['par'][2][0])]
					
		cmd = " ".join(cmd)
		
		l_y = y.tolist()
		l_x = x.tolist()
		
		model = svm_train(l_y, l_x, cmd)
		
		l_x_te = x_te.tolist()
		
		pred_label, stats, y_te = svm_predict(np.zeros(x_te.shape[0]), l_x_te, model)
		
	y_te = data_unscale.data_unscale(y_te, a, b, x.shape[1], cv_out)
	
	return y_te
		
		
		
		
		
		
		
		
		
		
		
		   