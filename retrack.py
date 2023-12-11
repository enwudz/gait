#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:42:03 2023

@author: iwoods
"""

import glob
import pandas as pd
import initializeClip
import autoTracker
import analyzeTrack
import analyzeSteps
import os
import shutil
import gaitFunctions

# retrack a movie but keep its steptracking data

# get excel files
print('\nGetting excel files . . . ')
xls = sorted(glob.glob('*.xlsx'))
xls = ['ea_14jul_tardigrade42_day2_013-098.xlsx'] # just testing one
track_threshold = 12

for xl in xls:
    
    # determine if this is a .mov or a .mp4 file
    fstem = xl.split('.')[0]
    mov = glob.glob(fstem + '.mov')
    mp4 = glob.glob(fstem + '.mp4')
    if len(mov) > 0:
        movie_file = fstem + '.mov'
    elif len(mp4) > 0:
        movie_file = fstem + '.mp4'
    else:
        exit('\n Cannot find a movie file for ' + xl)
    print(' ... movie file is ' + movie_file )
    
    # make a list to store steptracking sheets
    steptracking_sheets = []
    steptracking_sheet_data = []
    
    # see if there are steptracking sheets in this excel file
    xlworkbook = pd.ExcelFile(xl, engine='openpyxl')
    for sheet in xlworkbook.sheet_names:
        if 'steptracking' in sheet:
            # have steptracking data!
            print('Found ' + sheet + ' in ' + xl)
            df = pd.read_excel(xl, sheet_name=sheet)
            steptracking_sheets.append(sheet)
            steptracking_sheet_data.append(df)
    
    # delete the original excel file
    os.remove(xl)
    
    # run initialize clip
    initializeClip.main(movie_file)
    
    # run autoTracker
    autoTracker.main(movie_file, track_threshold)
    
    # run analyzeTrack
    analyzeTrack.main(movie_file)
    
    # add step tracking data if available
    if len(steptracking_sheets) > 0:
        for i, sheet in enumerate(steptracking_sheets):
            df = steptracking_sheet_data[i]
            with pd.ExcelWriter(xl, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
                df.to_excel(writer, index=False, sheet_name=sheet)
                
    # run analyzeSteps
    analyzeSteps.main(movie_file)


print('\n ... cleaning up ...')
fi = glob.glob('*first.png')
la = glob.glob('*last.png')
junk = fi + la
for j in junk: 
    os.remove(j)