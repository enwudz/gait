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
    segmentheight = 1.5
    segmentwidth = 1
    startpoint = [0,0]
    
    # body_buffer = 0.1
    # curve_buffer = 0.2
    # codes, verts = segment_blob(startpoint, body_buffer, curve_buffer, segmentheight, segmentwidth)
    
    codes,verts = cat_tail(startpoint, segmentwidth, segmentheight)
    
    # codes, verts, lefteye, righteye, ls_codes, ls_verts, rs_codes, rs_verts = tardigrade_head(startpoint, segmentwidth, segmentheight)
    bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
    ax.add_patch(bod)
    # ax.add_patch(lefteye)
    # ax.add_patch(righteye)
    # lstylet = mpatches.PathPatch(mpath.Path(ls_verts, ls_codes), ec='w', lw=3, fc = 'none')
    # ax.add_patch(lstylet)
    # rstylet = mpatches.PathPatch(mpath.Path(rs_verts, rs_codes), ec='w', lw=3, fc = 'none')
    # ax.add_patch(rstylet)

    ax.set_xlim([-2,2])
    ax.set_ylim([-2.5,2])
    ax.set_aspect('equal')
    plt.show()

def cat_tail(startpoint, segmentwidth, segmentheight):
    Path = mpath.Path
    
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.1 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.3 * segmentwidth, startpoint[1] - 0.56 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.1 * segmentwidth, startpoint[1] - 1.1 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.15 * segmentwidth, startpoint[1] - 1.3 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.15 * segmentwidth, startpoint[1] - 1.4 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0 * segmentwidth, startpoint[1] - 1.45 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.1 * segmentwidth, startpoint[1] - 1.35 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.25 * segmentwidth, startpoint[1] - 1.1 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.1 * segmentwidth, startpoint[1] - 0.56 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.2 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.CLOSEPOLY, [startpoint[0]+ 0 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        ])
    
    return codes, verts

def tardigrade_head(startpoint, segmentwidth, segmentheight):
    
    Path = mpath.Path
    eye_radius = 0.1 * segmentwidth
    
    headcodes, headverts = zip(*[
        (Path.MOVETO, [startpoint[0]+0 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        (Path.LINETO, [startpoint[0]-0.9 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]-1 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        (Path.LINETO, [startpoint[0]-1 * segmentwidth, startpoint[1]+0.08 * segmentheight ]),
        (Path.LINETO, [startpoint[0]-1 * segmentwidth, startpoint[1]+0.56 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]-0.95 * segmentwidth, startpoint[1]+0.64 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]-0.65 * segmentwidth, startpoint[1]+0.85 * segmentheight ]),
        (Path.LINETO, [startpoint[0]-0 * segmentwidth, startpoint[1]+0.90 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]+0.65 * segmentwidth, startpoint[1]+0.85 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]+0.95 * segmentwidth, startpoint[1]+0.64 * segmentheight ]),
        (Path.LINETO, [startpoint[0]+1 * segmentwidth, startpoint[1]+0.56 * segmentheight ]),
        (Path.LINETO, [startpoint[0]+1 * segmentwidth, startpoint[1]+0.08 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]+1 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        (Path.LINETO, [startpoint[0]+0.9 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        (Path.CLOSEPOLY, [startpoint[0]+0 * segmentwidth, startpoint[1]+0 * segmentheight ]),
        ])
    
    lefteye = mpatches.Circle([startpoint[0] - 0.6 * segmentwidth, startpoint[1] + 0.52 * segmentheight ], eye_radius, color = 'w')
    
    righteye = mpatches.Circle([startpoint[0] + 0.6 * segmentwidth, startpoint[1] + 0.52 * segmentheight ], eye_radius, color = 'w')
    
    ls_codes, ls_verts = zip(*[
        (Path.MOVETO, [startpoint[0]-0.1 * segmentwidth, startpoint[1]+0.8 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]-0.1 * segmentwidth, startpoint[1]+0.24 * segmentheight ]),
        (Path.LINETO, [startpoint[0]-0.6 * segmentwidth, startpoint[1]+0.12 * segmentheight ]),
        ])
    
    rs_codes, rs_verts = zip(*[
        (Path.MOVETO, [startpoint[0]+0.1 * segmentwidth, startpoint[1]+0.8 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]+0.1 * segmentwidth, startpoint[1]+0.24 * segmentheight ]),
        (Path.LINETO, [startpoint[0]+0.6 * segmentwidth, startpoint[1]+0.12 * segmentheight ]),
        ])
        

    
    return headcodes, headverts, lefteye, righteye , ls_codes, ls_verts, rs_codes, rs_verts
    

def cat_head(startpoint, segmentwidth, segmentheight):
    
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0],startpoint[1]]),
        (Path.LINETO, [startpoint[0] - 0.65 * segmentwidth,startpoint[1]]),
        (Path.CURVE3, [startpoint[0] - 0.88 * segmentwidth,startpoint[1] + 0.04 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 0.94 * segmentwidth,startpoint[1] + 0.21 * segmentheight]),
        (Path.CURVE3, [startpoint[0] - 1 * segmentwidth,startpoint[1] + 0.375 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 0.9 * segmentwidth,startpoint[1] + 0.6 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 1 * segmentwidth,startpoint[1] + 1 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 0.4 * segmentwidth,startpoint[1] + 0.79 * segmentheight]),
        (Path.CURVE3, [startpoint[0] ,startpoint[1] + 0.88 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 0.4 * segmentwidth,startpoint[1] + 0.79 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth,startpoint[1] + 1 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 0.9 * segmentwidth,startpoint[1] + 0.6 * segmentheight]),
        (Path.CURVE3, [startpoint[0] + 1 * segmentwidth,startpoint[1] + 0.375 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 0.94 * segmentwidth,startpoint[1] + 0.21 * segmentheight]),
        (Path.CURVE3, [startpoint[0] + 0.88 * segmentwidth,startpoint[1] + 0.04 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 0.65 * segmentwidth,startpoint[1]]),
        (Path.CLOSEPOLY, [startpoint[0],startpoint[1]]),
        ])
    
    return codes, verts

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