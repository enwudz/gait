#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:25:57 2023

@author: iwoods
    

CODE to make animation of leg state
    Empty matrix, defined by fps and time boundaries
    Each frame - 
    Shift matrix one step to left
    add current up/down state to right side
    Save frame:
    (grab movie frame?)
    (show box-walker too?)
    Show current leg states . . .
    show rear legs above, lateral legs below

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import gaitFunctions
import glob
import os
import scipy.signal

def main(movie_file):
    pass
    

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])

    main(movie_file)
    
    
