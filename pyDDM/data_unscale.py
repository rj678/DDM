#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:38:31 2018

@author: rishijumani
"""


import numpy as np


def data_unscale(x,a,b,yIND,cv_out):
	
	dmin = cv_out['dmin']
	dmax = cv_out['dmax']
	drange = dmax - dmin
	
	x = np.asarray(x)
	
	y = (x - a)/(b - a)*drange[0,yIND] + dmin[0,yIND]*np.ones(x.shape)
	
	return y
	
	