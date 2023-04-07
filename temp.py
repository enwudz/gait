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

    segmentwidth = 12
    segmentheight = 16
    startpoint = [0,0]
    fig, ax = plt.subplots(figsize=(8,6))
    codes, verts, larm_codes, larm_verts, rarm_codes, rarm_verts, head_codes, head_verts = human_torso(startpoint, segmentwidth, segmentheight)

    bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
    ax.add_patch(bod)

    larm = mpatches.PathPatch(mpath.Path(larm_verts, larm_codes), ec='w', lw=3, fc = 'none')
    ax.add_patch(larm)

    rarm = mpatches.PathPatch(mpath.Path(rarm_verts, rarm_codes), ec='w', lw=3, fc = 'none')
    ax.add_patch(rarm)

    head = mpatches.PathPatch(mpath.Path(head_verts, head_codes), ec='w', lw=3, fc = 'none')
    ax.add_patch(head)

    ax.set_xlim([-segmentwidth, segmentwidth])
    ax.set_ylim([0, segmentheight])

    ax.set_facecolor("steelblue") # slategray
    ax.set_aspect('equal')


    plt.show()


def human_torso(startpoint, segmentwidth, segmentheight):
    Path = mpath.Path
    
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0]+0 * segmentwidth, startpoint[1]+0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0]+0 * segmentwidth, startpoint[1]+ 0.06 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]- 0.09 * segmentwidth, startpoint[1]+ 0.06 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]- 0.25 * segmentwidth, startpoint[1]+ 0.12 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0.33 * segmentwidth, startpoint[1]+ 0.18 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]- 0.35 * segmentwidth, startpoint[1]+ 0.18 * segmentheight ]),
        (Path.CURVE4, [startpoint[0]- 0.75 * segmentwidth, startpoint[1]+ 0.25 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0.92 * segmentwidth, startpoint[1]+ 0.625 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0.83 * segmentwidth, startpoint[1]+ 0.94 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]- 0.71 * segmentwidth, startpoint[1]+ 0.98 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0.58 * segmentwidth, startpoint[1]+ 0.94 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0.62 * segmentwidth, startpoint[1]+ 0.6875 * segmentheight ]),
        (Path.CURVE3, [startpoint[0]- 0.25 * segmentwidth, startpoint[1]+ 0.75 * segmentheight ]),
        (Path.LINETO, [startpoint[0]- 0 * segmentwidth, startpoint[1]+ 0.75 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] + 0.25 * segmentwidth, startpoint[1]+ 0.75 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.62 * segmentwidth, startpoint[1]+ 0.6875 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.58 * segmentwidth, startpoint[1]+ 0.94 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] + 0.71 * segmentwidth, startpoint[1]+ 0.98 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.83 * segmentwidth, startpoint[1]+ 0.94 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.92 * segmentwidth, startpoint[1]+ 0.625 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.75 * segmentwidth, startpoint[1]+ 0.25 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.35 * segmentwidth, startpoint[1]+ 0.18 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.33 * segmentwidth, startpoint[1]+ 0.18 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.25 * segmentwidth, startpoint[1]+ 0.12 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.09 * segmentwidth, startpoint[1]+ 0.06 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0.06 * segmentheight ]),
        (Path.CLOSEPOLY, [startpoint[0] + 0 * segmentwidth, startpoint[1] +0.5 * segmentheight ])
    ])

    # arm lines
    larm_codes, larm_verts = zip(*[
        (Path.MOVETO, [startpoint[0] - 0.62 * segmentwidth, startpoint[1]+ 0.6875 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.63 * segmentwidth, startpoint[1]+ 0.625 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] - 0.62 * segmentwidth, startpoint[1]+ 0.52 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.5 * segmentwidth, startpoint[1]+ 0.5 * segmentheight ])
    ])

    rarm_codes, rarm_verts = zip(*[
        (Path.MOVETO, [startpoint[0] + 0.62 * segmentwidth, startpoint[1]+ 0.6875 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.63 * segmentwidth, startpoint[1]+ 0.625 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] + 0.62 * segmentwidth, startpoint[1]+ 0.52 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.5 * segmentwidth, startpoint[1]+ 0.5 * segmentheight ])
    ])

    # head line
    head_codes, head_verts = zip(*[
        (Path.MOVETO, [startpoint[0] - 0.33 * segmentwidth, startpoint[1]+ 0.185 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.34 * segmentwidth, startpoint[1]+ 0.25 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.33 * segmentwidth, startpoint[1]+ 0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.22 * segmentwidth, startpoint[1]+ 0.58 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] - 0.08 * segmentwidth, startpoint[1]+ 0.64 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0 * segmentwidth, startpoint[1]+ 0.643 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] + 0.08 * segmentwidth, startpoint[1]+ 0.64 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.22 * segmentwidth, startpoint[1]+ 0.58 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.33 * segmentwidth, startpoint[1]+ 0.5 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.34 * segmentwidth, startpoint[1]+ 0.25 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.33 * segmentwidth, startpoint[1]+ 0.185 * segmentheight ])
    ])

    return codes, verts, larm_codes, larm_verts, rarm_codes, rarm_verts, head_codes, head_verts

main()    