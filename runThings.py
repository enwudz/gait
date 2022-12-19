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
import gait_analysis
import sys

def main(movie_file):
    initializeClip.main(movie_file)
    trackCritter.main(movie_file)
    analyzePath.main(movie_file)

if __name__== "__main__":

    if len(sys.argv) > 1:
        selection =sys.argv[1]
        if 'mov' in selection:
            movie_file = sys.argv[1]
        elif selection.isin['all','a']:
            movie_files = glob.glob('*.mov')
            for movie_file in movie_files:
                main(movie_file)
        else:
            print('Please provide a .mov file or type a or all')
    else:
       movie_file = gait_analysis.select_movie_file()
       
    print('Movie is ' + movie_file)

    main(movie_file)
