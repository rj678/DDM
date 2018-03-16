#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 20:06:37 2018

@author: rishijumani
"""

"""
returns : Percent bias (PBIAS)
          mean absolute error (MAE)
          Nash-Sutcliff coefficient (NSE)
"""

import numpy as np

def performance_assess(obs, comp):
	y = np.zeros(3)
	y[0] = np.mean(obs - comp)/np.mean(obs)*100
	y[1] = np.mean(np.abs(obs - comp))
	y[2] = 1 - np.mean((obs - comp)**2)/np.var(obs)
	
	return y
	