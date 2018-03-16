#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:29:33 2018

@author: rishijumani
"""

import assess_dist

import estimate_PI

import pickle
import scipy.io as spio

mat = spio.loadmat('test.mat', squeeze_me=True)
#x_te = mat['x_te']

with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
    x, yobs, ysim, opt, cv_out, x_te = pickle.load(f)
	
import scipy.io as spio

mat = spio.loadmat('test.mat', squeeze_me=True)
x_te = mat['x_te']

	
#with open('dist.pkl') as f:  # Python 3: open(..., 'rb')
#    res = pickle.load(f)

scores, pi = estimate_PI.estimate_PI(x, yobs - ysim, opt, cv_out, x_te)

#assess_dist.assess_dist(res)