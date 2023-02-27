#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:12:11 2023

@author: iwoods
"""

import sys
import os

def main(searchterm, fps, outfile):
    # needs ffmpeg installed
    cmd1 = "ffmpeg -f image2 -r "
    # fps
    cmd2 = " -pattern_type glob -i '"
    # searchterm
    cmd3 = "' -pix_fmt yuv420p -crf 20 "
    # outfile
    
    cmd = cmd1 + str(fps) + cmd2 + searchterm + cmd3 + outfile
    print(cmd)
    try:
        os.system(cmd)
    except:
        usage()
    
def usage():
    print("Usage: python makeMovieFromImages.py 'searchterm' fps outfile")
    
if __name__== "__main__":

    print(len(sys.argv))
    if len(sys.argv) == 4:
        searchterm = sys.argv[1]
        fps = sys.argv[2]
        outfile = sys.argv[3]
        main(searchterm, fps, outfile)
    else:
        usage()
        
