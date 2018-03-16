#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 20:36:18 2018

@author: rishijumani
"""

import numpy as np

def data_scale(x, a, b):
	xmax = np.max(x, axis=0)
	xmax = xmax.reshape(1,xmax.shape[0])
	xmin = np.min(x, axis=0)
	xmin = xmin.reshape(1,xmin.shape[0])
	drange = xmax - xmin
	y = (x - np.ones((x.shape[0],1))*xmin)*(b - a)/  \
		(np.ones((x.shape[0],1))*drange) + a*np.ones(np.shape(x))
		
	return y, xmin, xmax
	
