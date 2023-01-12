#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:27:49 2023

@author: iwoods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import gaitFunctions
import sys


def main(combined_file):

    # check if output file exists ... ask if add to it or make a new one
    # if new one, or none exists, ask what to call the output file

    # load in data    
    try:
        track_df = pd.read_excel(combined_file, sheet_name='pathtracking', index_col=None)
        pathsummary_df = pd.read_excel(combined_file, sheet_name='path_summaries', index_col=None)
        have_track = True
    except:
        track_df = pd.DataFrame()
        pathsummary_df = pd.DataFrame
        have_track = False
        
    try:
        step_df = pd.read_excel(combined_file, sheet_name='step_timing', index_col=None)
        have_steps = True
    except:
        step_df = pd.DataFrame()
        have_steps = False
    
    try:
        gait_df = pd.read_excel(combined_file, sheet_name='gait_styles', index_col=None)
        have_gait = True
    except:
        gait_df = pd.DataFrame()
        have_gait = False    
   
    if have_track:
        # go through clips and get: total distance, total time, total turns, total bearings
                
        # areas => save to summary as 'median_area'
        
        # lengths => save to summary as 'median_length'
        
        # speed => save to summary as 'speed_mm_per_sec'
        #   need total distance and total time across clips
        
        # save it to summary file 
        # (if this individual already exists ... we can drop it from the dataframe)
        
        # ask if we should plot the tracking data
        pass
        
        

def select_combined_file():
    combined_data_files = sorted(glob.glob('*_combined.xlsx'))
    if len(combined_data_files) > 0:
        combined_file = gaitFunctions.selectOneFromList(combined_data_files)
    else:
        combined_file = ''
        print('Cannot find a combined data file - do you have one here?')
    return combined_file

if __name__== "__main__":

    if len(sys.argv) > 1:
        combined_file = sys.argv[1]
    else:
       combined_file = select_combined_file()
       
    print('Reading in ' + combined_file)

    main(combined_file)