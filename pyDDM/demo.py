#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:51:11 2018

@author: rishijumani


to save workspace:
	
import pickle
	
with open('objs.pkl', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([x, yobs, ysim, opt, cv_out, x_te], f)


to load workspace:
	
with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    x, yobs, ysim, opt, cv_out, x_te = pickle.load(f)


todo:
	
	
check the order of svm_predict op args





"""

import scipy.io as spio
import numpy as np

# import the main DDM-UQ module
import ddmuq

# load sample MATLAB dataset into dictionary
mat = spio.loadmat('test.mat', squeeze_me=True)

"""
 get input data from dictionary

  x       n by p matrix of input data for training, n is the number of data points
	             and p is the dimension of input variables.

  yobs    n by 1 vector of historical observations used for training
  ysim    n by 1 vector of model simulated equivalents used for training 
  x_te    m by p matrix of testing input data
  yobs_te m by 1 vector of observations used for testing
  ysim_te m by 1 vector of model simulation used for testing
  opt     a MATLAB structure converted to a dictionary with the following keys
          opt.ddm: 1=QRF 2=SVR; more options to be added
          opt.nfold:  cross validation (CV) # folds, active only if opt.ddm=2
          opt.cv_par1: (active only if opt.ddm=2) parameter (e.g. g)
          	    values used in nfold cross validation
          opt.cv_par2: (active only if opt.ddm=2) parameter (e.g. epsilon)
              values used in nfold cross validation
          opt.cl: desired confidence interval of PI, e.g. 0.9
"""


x = mat['x']
yobs = mat['yobs']
ysim = mat['ysim']
x_te = mat['x_te']
yobs_te = mat['yobs_te']
ysim_te = mat['ysim_te']

opt_d = mat['opt']
opt_l = opt_d.tolist()
opt_n = opt_d.dtype.names
opt = dict(zip(opt_n, opt_l))

"""
 OUTPUT DATA:

  ybar    DDMs updated model simulation
  pi      DDMs estimated confidence interval
  perf    perf.mean: PBIAS, MAE and NSE before and after DDMs updating
          perf.PICP: coverage probability of estimated PI or CI
  ybar_te DDMs updated model prediction
  pi_te   DDMs estimated prediction interval
  perf_te Same as perf, except for testing period
  picp    Actual coverage probability of of estimated confidence interval
"""



# call DDM-UQ in training mode
#cv_out = ddmuq.ddmuq(x, yobs, ysim, opt)



# call DDM-UQ in forecast mode. 
# This mode only outputs DDM updated prediction and PI, 
# no performance assessment because no validation observation is available
cv_out = ddmuq.ddmuq(x, yobs, ysim, opt, x_te, ysim_te)



