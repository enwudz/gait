#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:14:59 2023

@author: iwoods
"""

import numpy as np

def minDiffVectors(vec1, vec2):
    
    if len(vec1) == len(vec2):
        A = vec1
        B = vec2
    elif len(vec2) > len(vec1):
        A = vec2
        B = vec1
    else:
        A = vec1
        B = vec2
        
    avg_diff = 100
    minDiffVecs = []
   
    # no offset
   
    
vec1 = np.array([1,2,3,4,5])
vec2 = np.array([1,2,3,4,5,6])
    
minDiffVectors(vec1, vec2)