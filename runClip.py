#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:32:38 2022

@author: iwoods
"""

import glob
import initializeClip
import trackCritter
import analyzeTrack
import plotClip
import gaitFunctions
import frameStepper
import analyzeSteps
import plotSteps
import sys

def main(movie_file): 
    # can 'comment' individual steps on or off
    # <=== put a # before a line to comment it off
    
    ## make an excel file for a clip
    # initializeClip.main(movie_file)
    
    ## automated path tracking, including speed, turns, stops
    ## if tracking wonky at 12, try 25, eg: (movie_file, 25, True) 
    # trackCritter.main(movie_file, 12, True) # True is show tracking
    # analyzeTrack.main(movie_file)
    # plotClip.main(movie_file)
    
    ## step-by-step timing
    # frameStepper.main(movie_file)
    analyzeSteps.main(movie_file)
    # plotClip.main(movie_file)


if __name__== "__main__":

    if len(sys.argv) > 1:
        selection = sys.argv[1]
        
        # can designate ONE movie in command
        if 'mov' in selection:
            movie_file = sys.argv[1]
            
        # can also designate 'all' movies
        elif selection in ['all','a']:
            movie_files = sorted(glob.glob('*.mov'))
            for movie_file in movie_files:
                main(movie_file)
                sys.exit('Finished processing all clips')
        
        # if nothing given in command, list movie files and ask which one to do
        else:
            movie_file = gaitFunctions.select_movie_file()
    else:
       movie_file = gaitFunctions.select_movie_file()

    if len(movie_file) > 0:
        print('Selected movie is ' + movie_file)
        main(movie_file)
    else:
        print('Cannot find a .mov file . . .')
