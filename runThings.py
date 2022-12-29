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
import plotPath
import gaitFunctions
import frameStepper
import analyzeSteps
import sys

def main(movie_file): # I am on 29 and 30
    # initializeClip.main(movie_file)
    # trackCritter.main(movie_file, 25, True) # try (movie_file, 12 OR 25) if tracking wonky
    # analyzePath.main(movie_file)
    # plotPath.main(movie_file,'track')
    # plotPath.main(movie_file,'time')
    frameStepper.main(movie_file)
    # analyzeSteps.main(movie_file)
    

if __name__== "__main__":

    if len(sys.argv) > 1:
        selection =sys.argv[1]
        if 'mov' in selection:
            movie_file = sys.argv[1]
        elif selection in ['all','a']:
            movie_files = sorted(glob.glob('*.mov'))
            for movie_file in movie_files:
                main(movie_file)
        else:
            print('Please provide a .mov file or type "a" or "all"')
    else:
       movie_file = gaitFunctions.select_movie_file()
       print('Movie is ' + movie_file)

    main(movie_file)
