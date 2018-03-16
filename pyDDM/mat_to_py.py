#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:32:22 2018

@author: rishijumani
"""

import scipy.io as spio

#import json

mat = spio.loadmat('test.mat', squeeze_me=True)

#f = open('workfile', 'w')

#json.dump(mat, f)