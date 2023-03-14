#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:58:21 2023

@author: iwoods

animation of legs

(To make simulation)
From gait cycle, duty factor, anterior offsets, opposite offsets
    make frame times based on 30 fps for N gait cycles
	make list of up times and down times for all legs

(For real data and simulation)
From list of up times and down times for all legs ... and frame times
    determine extent of swing at each frame
    record leg state at each frame	

(Add head)
From axis, number of segments, segment width
	add a head to axis

(Add tail)
From axis, number of segments)
	add a tail to axis


"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import gaitFunctions
from matplotlib.animation import FuncAnimation

def main():

    arr = np.linspace(0,1,50)
    swingvals = np.hstack((arr,np.flip(arr)))
    
    legs = ['L1','R1','L2','R2','L3','R3','L4','R4']
    swings = [1, 1, 0.6, 0.6, 0.3, 0.3, 0, 0]
    swingextents = dict(zip(legs,swings))
    legstates = {}
    for leg in legs:
        legstates[leg] = 'up'
    
    f,ax = plt.subplots()
    ax = drawLegs(ax, swingextents, legstates)
    
    # have swingextents = dictionary of swing extents
    # keys = leg names, values = array of swing extents for each leg
    
    # have legstates = a dictionary of leg states (up or down)
    # keys = leg names, values = array of up or downs
    
    # ani = FuncAnimation(fig, animate, frames=len(swingextents), 
    #                     interval=33, repeat=True, fargs=[ax, swingextents, legstates]) 
    
    plt.show()

def animate(i, ax, swingextents, legstates):
        
    
    
    return


def rightSegmentPatch(midright, body_buffer, segmentheight, segmentwidth):
    
    midright[0] += body_buffer
    curve = 0.2
    xstart = midright[0]
    ystart = midright[1]
    segmentwidth += body_buffer
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, midright), # get to start
        (Path.LINETO, [xstart, ystart + (1-curve) * segmentheight/2]), # line to beginning of upper right curve
        # (Path.CURVE4, [xstart + curve * segmentwidth, ystart + segmentheight/2]), # upper right curve ... but need two more points to make curve
        (Path.LINETO, [xstart - segmentwidth, ystart + segmentheight/2]), # line to upper left corner
        (Path.LINETO, [xstart - segmentwidth, ystart - segmentheight/2]), # line to lower left corner
        (Path.LINETO, [xstart, ystart - (1-curve) * segmentheight/2]), # line to beginning of lower left curve
        # (Path.CURVE4, [xstart + curve * segmentwidth, ystart - (1-curve) * segmentheight/2]), # lower left curve ... but need two more points to make curve
        (Path.CLOSEPOLY, midright) # line to beginning
        ])
    
    return codes, verts

def leftSegmentPatch(midleft, body_buffer, segmentheight, segmentwidth):
    
    midleft[0] -= body_buffer
    curve = 0.2
    xstart = midleft[0]
    ystart = midleft[1]
    segmentwidth += body_buffer
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, midleft), # get to start
        (Path.LINETO, [xstart, ystart + (1-curve) * segmentheight/2]), # line to beginning of upper left curve
        # (Path.CURVE4, [xstart + curve * segmentwidth, ystart + segmentheight/2]), # upper left curve ... but need two more points to make curve
        (Path.LINETO, [xstart + segmentwidth, ystart + segmentheight/2]), # line to upper right corner
        (Path.LINETO, [xstart + segmentwidth, ystart - segmentheight/2]), # line to lower right corner
        (Path.LINETO, [xstart, ystart - (1-curve) * segmentheight/2]), # line to beginning of lower left curve
        # (Path.CURVE4, [xstart + curve * segmentwidth, ystart - (1-curve) * segmentheight/2]), # lower left curve ... but need two more points to make curve
        (Path.CLOSEPOLY, midleft) # line to beginning
        ])
    
    return codes, verts

def arcColor(magnitude, blackwhite):
    
    '''
    Parameters
    ----------
    magnitude : floating point decimal between 0 and 1
        how dark or light do we want the shade.
    blackwhite : string
        'black' or 'white'.

    Returns
    -------
    shade : tuple
        color (0,0,0 is black, (1,1,1) is white

    '''
    
    buffer = 0.8
    x = np.linspace(buffer, np.pi-buffer, 100)
    y = np.sin(x)
    c = np.array([1,1,1])
    
    pos = int(magnitude*100)-1
    shade = y[pos] * c
    
    if blackwhite == 'black':
        shade = np.array([1-x for x in shade])
    
    return shade

def swingAngle(swingextent):
    # swingextent is a floating point decimal between 0 and 1
    min_angle = 45
    max_angle = 180 - min_angle
    swing_index = int(swingextent*100)
    angles = np.linspace(max_angle, min_angle, 101)
    return angles[swing_index]

def drawLegs(ax, swingextents, legstates):
    '''
    
    Parameters
    ----------
    ax : matplotlib axis object
    swingextents : dictionary
        keys = names of legs
        values = floating point decimal between 0 and 1
        1 = anterior-most position of leg, 0 = posterior-most position of leg
    legstates : dictionary
        keys = names of legs    
        values = state of leg 'up' for swing, 'down' for stance.

    Returns
    -------
    ax

    '''
    
    leg_thickness = 0.6 # height
    leg_length = 1.6 # width
    
    segment_width = 0.8 * leg_length
    segment_height = 2.2 * leg_length
    
    body_buffer = 0.2 * leg_length
    
    all_legs = gaitFunctions.get_leg_list(10)
    legs = [x for x in all_legs if x in swingextents.keys()]

    if len(legs) % 2 == 0:
        num_rows = int(len(legs) / 2)
        num_cols = int(len(legs) / num_rows)
    else:
        num_rows = 1
        num_cols = 1 
    
    body_width = num_cols * (leg_length + segment_width)
    body_length = num_rows * segment_height
    
    # get leg points from number of legs
    # build up from front left
    leg_points = {}
    start_x = 0
    start_y = 0
    counter = 0
    for row in range(num_rows):
        for col in range(num_cols):
            
            # add column offset to starting x point
            this_x = start_x + col * 2 * segment_width
            
            # add row offset to starting y point
            this_y = start_y - row * segment_height
            
            # get point for this leg
            leg_points[legs[counter]] = [this_x, this_y]

            counter += 1
    
    # go through legs and plot
    for leg in legs:
    
        # get color and angle for this leg
        leg_color = arcColor(swingextents[leg], legstates[leg])
        leg_angle = swingAngle(swingextents[leg])
        leg_point = leg_points[leg]
        bod_point = leg_points[leg]
        
        point_of_rotation = np.array([leg_point[0], leg_point[1] + leg_thickness/2])
        
        if 'L' in leg:
            # leg_point[0] = leg_point[0] + 0.2 * leg_length
            rec = mpatches.Rectangle(leg_point, width=leg_length, height=leg_thickness, color = leg_color,
                                transform=Affine2D().rotate_deg_around(*point_of_rotation, 90+leg_angle)+ax.transData)
            codes,verts = leftSegmentPatch(bod_point, body_buffer, segment_height, segment_width)
        else:
            # leg_point[0] = leg_point[0] - 0.2 * leg_length
            rec = mpatches.Rectangle(leg_point, width=leg_length, height=leg_thickness, color = leg_color,
                                transform=Affine2D().rotate_deg_around(*point_of_rotation, 90-leg_angle)+ax.transData)
            codes,verts = rightSegmentPatch(bod_point, body_buffer, segment_height, segment_width)
        
        
        # plt.scatter(point_of_rotation[0],point_of_rotation[1], c='r', s=50) # check leg point
        bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
        ax.add_patch(rec)
        ax.add_patch(bod)
    
    # set axis limits
    ax.set_aspect('equal')
    ax.set_xlim([-leg_length * 1.2, 1.2 * (body_width - leg_length * 1.2)])
    ax.set_ylim([-10,1])
    ax.set_ylim([ segment_height/2 - body_length, segment_height/2 ] )
    
    # # set axis orientation
    # if leftright == 'right':
    #     ax.invert_xaxis()
    
    # # set background color
    ax.set_facecolor("steelblue") # slategray
    
    # # clear the frame and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return ax

if __name__ == '__main__':
    main()




