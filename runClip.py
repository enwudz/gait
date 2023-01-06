#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:32:38 2022

@author: iwoods
"""

import glob
import initializeClip
import trackCritter
import analyzePath
import plotClip
import gaitFunctions
import frameStepper
import analyzeSteps
import plotSteps
import sys

def main(movie_file): 
    # can 'comment' individual steps on or off
    # <=== put a # before a line to comment it off
    # initializeClip.main(movie_file)
    # ## ==>  try (movie_file, 12 OR 25) if tracking wonky; True is show tracking
    # trackCritter.main(movie_file, 12, True) 
    # analyzePath.main(movie_file)
    # plotClip.main(movie_file,'track')
    # plotClip.main(movie_file,'speed')
    # frameStepper.main(movie_file)
    # analyzeSteps.main(movie_file)
    plotClip.main(movie_file,'steps')
    # plotClip.main(movie_file,'legs')

if __name__== "__main__":

    if len(sys.argv) > 1:
        selection =sys.argv[1]
        
        # can designate ONE movie in command
        if 'mov' in selection:
            movie_file = sys.argv[1]
            
        # can also designate 'all' movies
        elif selection in ['all','a']:
            movie_files = sorted(glob.glob('*.mov'))
            for movie_file in movie_files:
                main(movie_file)
        
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
