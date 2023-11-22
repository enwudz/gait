#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:48:23 2023

@author: iwoods
"""

import glob
import gaitFunctions
import pandas as pd
import initializeClip
import autoTracker
import analyzeTrack
import analyzeSteps
import os
import sys

'''
legacySteptracking take existing steptracking and break up the data into cruise bouts . . . 
	Delete excel file
	Run initializeClip
	Run autoTracker
	Run analyzeTrack
	Save existing steptracking data to stepALLtracking
	For each cruise bout
		Save steptracking data to appropriate sheet
	Run analyzeSteps
'''

def main(movie_file, retrack = True):
    
    update_excel = False
    
    fstem = movie_file.split('.')[0]
    excel_file = fstem + '.xlsx'
    
    # see if there are steptracking sheets in this excel file
    xlworkbook = pd.ExcelFile(excel_file, engine='openpyxl')
    if 'steptracking' in xlworkbook.sheet_names:
        update_excel = True
        legacy_step_df = pd.read_excel(excel_file, sheet_name='steptracking')
            
    if update_excel:
        
        # do we want to retrack?
        if retrack:
            
            # remove this excel file
            os.remove(excel_file)
            
            # initialize Clip
            initializeClip.main(movie_file)
            
            # auto tracker
            autoTracker.main(movie_file, 12, True)
            
            # analyze track
            analyzeTrack.main(movie_file)
            
        # retracking finished ... save legacy step data
        with pd.ExcelWriter(excel_file, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            legacy_step_df.to_excel(writer, index=False, sheet_name='stepALLtracking')
    
        # Make new sheets for bout step data
        legacy_leg_states = legacy_step_df.leg_state.values
        legacy_times = legacy_step_df.times.values
        
        # load the path_stats page for this movie
        path_stats = gaitFunctions.loadPathStats(movie_file)
        
        # print informmation about cruising bouts for this movie
        print('...this clip has ' + str(path_stats['# cruise bouts']) + ' bouts of cruising:')
        
        cruise_bouts = path_stats['cruise bout timing'].split(';')
        for bout in cruise_bouts:
                print('   ' + bout)
        
        # make a steptracking sheet for each bout
        print('Making a steptracking sheet for each cruise bout ... ')
        for bout in cruise_bouts:
            boutstart = float(bout.split('-')[0].replace(' ',''))
            boutend = float(bout.split('-')[1].replace(' ',''))
            time_string = str(int(boutstart)) + '-' + str(int(boutend))

            steptracking_sheetname = 'steptracking_' + time_string
            
            # working
            bout_step_times = []
            for i, state in enumerate(legacy_leg_states):
                step_times = [float(x) for x in legacy_times[i].split()]
                after_start = [x for x in step_times if x >= boutstart]
                before_end = [x for x in after_start if x <= boutend]
                bout_step_times.append(' '.join([str(x) for x in before_end]))
                
            bout_df = pd.DataFrame({'leg_state':legacy_leg_states, 'times':bout_step_times})
            with pd.ExcelWriter(excel_file, if_sheet_exists='replace', engine='openpyxl', mode='a') as writer:
                bout_df.to_excel(writer, index=False, sheet_name=steptracking_sheetname)
        
    # analyze steps
    analyzeSteps.main(movie_file)
    
    # clean up trash
    gaitFunctions.cleanUpTrash(movie_file)

if __name__== "__main__":

    if len(sys.argv) > 1:
        
        if sys.argv[1] == 'all':
            movs = glob.glob('*.mov')
            mp4s = glob.glob('*.mp4')
            movie_files = sorted(movs + mp4s)
            for movie_file in movie_files:
                main(movie_file)
        else:
            movie_file = sys.argv[1]
    
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
    
    main(movie_file)