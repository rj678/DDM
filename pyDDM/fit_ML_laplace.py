#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:17:01 2018

@author: rishijumani
"""

import numpy as np
import matplotlib.pyplot as plt



def fit_ML_laplace(x):
	
	N = x.shape[0]
	u = np.sum(x)/N
	b = np.sum(np.abs(x - u))/N
	CRB_b = b**2/N
	n, x_c = np.histogram(x, bins=100)
	
	# get the centers of the bins - simplify this
	
	for i,_ in enumerate(x_c):		
		if i < x_c.shape[0]-1:
			x_c[i] = (x_c[i] + x_c[i+1])/2
			
	x_c = x_c[:-1]
	
	n = n/np.sum(n*np.abs(x_c[1] - x_c[0]))
	y = 1/(2*b)*np.exp(-np.abs(x_c - u)/b)
	RMS = np.sqrt((y - n)*((y - n).T)/(x_c[1] - x_c[0])**2/(x_c.shape[0]))
	
	
	result = {'u':u, 'b':b, 'CRB_b':CRB_b, 'RMS':RMS}
	
	return result

	
	