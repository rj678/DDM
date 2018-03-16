#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 20:34:14 2018

@author: rishijumani

"""

import numpy as np
import copy
import matplotlib.pyplot as plt


import data_scale
import performance_assess

# add libsvm to path
import sys
sys.path.append('/Volumes/A_2TB/Work/Papers_Code/bin/libsvm-3.22/python')

from svmutil import *


def ddm_train(x, y, opt):
	
	# scale to (0,1) range
	
	a = 0.01
	b = 0.99
	y = y.reshape(y.shape[0],1)
	dat = np.hstack((x, y))
	dat, dmin, dmax = data_scale.data_scale(dat, a, b)
	
	x = dat[:,:-1]
	y = dat[:,-1]
	
	y = y.reshape(y.shape[0],1)
	
	if opt['ddm'] == 2:
		nfold = opt['nfold']
		C = np.maximum(np.abs(np.mean(y + 3*np.std(y))), \
						 np.abs(np.mean(y - 3*np.std(y))))
		cvid = np.random.permutation(np.shape(x)[0])
		cvid = np.mod(cvid, nfold)
		ecv_tr = np.zeros((opt['cv_par1'].shape[0], opt['cv_par2'].shape[0], 3, nfold))
		ecv_te = copy.deepcopy(ecv_tr)
		
		for cv in range(nfold):
			te_id = cvid == cv
			tr_id = np.logical_not(te_id)
			cv_xtr = x[tr_id, :]
			cv_xte = x[te_id, :]
			cv_ytr = y[tr_id, :]
			cv_yte = y[te_id, :]
			
			cv_ytr = cv_ytr.reshape(cv_ytr.shape[0])
			cv_yte = cv_yte.reshape(cv_yte.shape[0])
			
			
			for i in range(opt['cv_par1'].shape[0]):
				for j in range(opt['cv_par2'].shape[0]):
					cmd = ['-q -s 3 -t 2 -c', str(C), '-g', str(opt['cv_par1'][i]),
								 '-p', str(opt['cv_par2'][j])]
					
					cmd = " ".join(cmd)
					
					l_cv_ytr = cv_ytr.tolist()
					l_cv_xtr = cv_xtr.tolist()
					l_cv_yte = cv_yte.tolist()
					l_cv_xte = cv_xte.tolist()

					model = svm_train(l_cv_ytr, l_cv_xtr, cmd)
					y1,accu1,prob = svm_predict(l_cv_ytr,l_cv_xtr,model)
					y2,accu2,prob = svm_predict(l_cv_yte,l_cv_xte,model)
					ecv_tr[i,j,:,cv] = performance_assess.performance_assess(cv_ytr,y1)
					ecv_te[i,j,:,cv] = performance_assess.performance_assess(cv_yte,y2);
					
					
		# plot CV results
		
		fig, ax = plt.subplots(nrows=3, ncols=1)
		
		for i in range(3):
			
			ax[i].contourf(np.mean(ecv_te[:,:,i,:], axis=0))
			if i == 0:
				ax[i].set_title('PBIAS')
			elif i ==1:
				ax[i].set_title('MAE')
			else:
				ax[i].set_title('NSE')
				ax[i].set_xlabel(r'$\varepsilon$')
				ax[i].set_ylabel('g')
				
			
		plt.show()
		
		# choose best parameters
		
		ecv_tmp = np.mean(ecv_te[:,:,2,:], axis=2)
		rows, cols = np.where(ecv_tmp == np.min(np.min(ecv_tmp)))
		par_cv = [C, opt['cv_par1'][rows], opt['cv_par2'][cols]]
		cv_out={}
		cv_out['dmin'] = dmin
		cv_out['dmax'] = dmax
		cv_out['par'] = par_cv 
		
	return cv_out
				
			
			
			
			
			
			
			
			


		
	