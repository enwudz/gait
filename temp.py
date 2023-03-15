# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import os
# import shutil
# import glob
# import pandas as pd
# import gaitFunctions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches


def main():
    
    fig,ax = plt.subplots()
    body_buffer = 0.1
    curve_buffer = 0.2
    segmentheight = 1
    segmentwidth = 1
    startpoint = [-3,0]
    codes, verts = segment_blob(startpoint, body_buffer, curve_buffer, segmentheight, segmentwidth)
    bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
    ax.add_patch(bod)
    ax.set_xlim([-5,1])
    ax.set_ylim([-2,2])
    plt.show()

def segment_blob(midright, body_buffer, curve_buffer, segmentheight, segmentwidth):
    
    curve_offset = curve_buffer * segmentheight
    body_offset = body_buffer * segmentwidth
    midright[0] += body_offset
    xstart = midright[0]
    ystart = midright[1]
    segmentwidth += body_buffer
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, midright), # get to start
        (Path.LINETO, [xstart, ystart + segmentheight/2 - curve_offset]), 
        (Path.CURVE3, [xstart, ystart + segmentheight/2]),
        (Path.LINETO, [xstart - curve_offset, ystart + segmentheight/2]),
        (Path.LINETO, [xstart - segmentwidth/2, ystart + segmentheight/2]), 
        (Path.LINETO, [xstart - segmentwidth/2, ystart - segmentheight/2]),
        (Path.LINETO, [xstart - curve_offset, ystart - segmentheight/2]),
        (Path.CURVE3, [xstart, ystart - segmentheight/2]),
        (Path.LINETO, [xstart, ystart - segmentheight/2 + curve_offset]),
        (Path.CLOSEPOLY, midright) # line to beginning
        ])
    
    return codes, verts

main()