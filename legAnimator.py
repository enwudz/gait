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

    ## ==> get up / down times for legs
    ## either from framestepper in an experiment excel file
    # up_down_times, frame_times = load_movie_steps()
    
    ## OR make simulated data based on step parameters
    up_down_times, frame_times = load_simulated_steps()
    # print(up_down_times)
    
    ## ==> want 2 dictionaries:
    ## legstates = a dictionary of leg states (up or down)
    ##     keys = leg names, values = list of up or downs
    ## legangles = a dictionary of swing extents (backward = 0, forward = 1)
    ##     keys = leg names, values = array of swing extents
    legstates, legangles = get_leg_swings(up_down_times, frame_times)

    ## make basic animation
    basic_animation(legangles, legstates, False) # True to save      

def load_simulated_steps():
    ## define step parameters
    simulation = {}
    simulation['num_legs'] = 4
    simulation['num_cycles'] = 10
    simulation['gait_cycle'] = 0.43 # in seconds
    simulation['duty_factor'] = 0.44 # in fraction of gait cycle
    simulation['opposite_offset'] = 0.5 # in fraction of gait cycle
    simulation['anterior_offset'] = 0.5  # in fraction of gait cycle
    simulation['fps'] = 30
    up_down_times, frame_times = simulate_steps(simulation)
    return up_down_times, frame_times

def basic_animation(legangles, legstates, save_animation = False):    
    fig, ax = plt.subplots(figsize=(7,8))
    ani = FuncAnimation(fig, animate_steps, frames=len(legangles['L1']), 
                        interval=33, repeat=False, fargs=[ax, legangles, legstates]) 
    
    if save_animation:
        ani.save('animation.mp4')
    plt.show()
    
def animate_steps(i, ax, legangles, legstates):
        
    legs = list(legangles.keys())
    
    ax.clear()
    
    legstate = {}
    swingextent = {}
    
    for leg in legs:
        legstate[leg] = legstates[leg][i]
        swingextent[leg] = legangles[leg][i]
    
    ax = drawLegs(ax, swingextent, legstate)
    
    return

def get_leg_swings(up_down_times, frame_times):    
    ## get legs, with specified leg order
    num_legs = len(up_down_times)
    legs = gaitFunctions.get_leg_list(num_legs)
    legs = [x for x in legs if x in up_down_times.keys()]
    
    ## get leg matrix
    leg_matrix = gaitFunctions.make_leg_matrix(legs, up_down_times, frame_times)
    
    ## get leg states (up or down) and leg angles (forward or backward extent) from leg_matrix
    legstates = {}
    legangles = {}
    for i, leg in enumerate(legs):
        leg_binary = leg_matrix[i,:]
        legstates[leg] = ['up' if x == 1 else 'down' for x in leg_binary]

        # get indices of runs of zeros (stances) and ones (swings)
        one_runs = gaitFunctions.one_runs(leg_binary)
        zero_runs = gaitFunctions.zero_runs(leg_binary)
        
        angles = np.zeros(len(leg_binary))
        # for swings ... legangle is 0 at beginning of swing and 1 at end of swing
        for run in one_runs: # swings!
            len_run = run[1] - run[0]
            vals = np.linspace(0,1,len_run)
            angles[run[0]:run[1]] = vals
        
        # for stance ... legangle is 1 at beginning of stance and 0 at end of stance
        for run in zero_runs: # stances!
            len_run = run[1] - run[0]
            vals = np.linspace(1,0,len_run)
            angles[run[0]:run[1]] = vals
            
        legangles[leg] = angles
        
    return legstates, legangles

def load_movie_steps():
    movie_file = gaitFunctions.selectFile(['mp4','mov'])
    frame_times = gaitFunctions.getFrameTimes(movie_file)
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file.split('.')[0]) 
    mov_data, excel_filename = gaitFunctions.loadUpDownData(excel_filename)
    up_down_times, last_event = gaitFunctions.getUpDownTimes(mov_data)
    return up_down_times, frame_times

def rightSegmentPatch(midright, body_buffer, curve_buffer, segmentheight, segmentwidth):
    
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
        (Path.LINETO, [xstart - segmentwidth - body_offset, ystart + segmentheight/2]), 
        (Path.LINETO, [xstart - segmentwidth - body_offset, ystart - segmentheight/2]),
        (Path.LINETO, [xstart - curve_offset, ystart - segmentheight/2]),
        (Path.CURVE3, [xstart, ystart - segmentheight/2]),
        (Path.LINETO, [xstart, ystart - segmentheight/2 + curve_offset]),
        (Path.CLOSEPOLY, midright) # line to beginning
        ])
    
    return codes, verts


def leftSegmentPatch(midleft, body_buffer, curve_buffer, segmentheight, segmentwidth):
    
    curve_offset = curve_buffer * segmentheight
    body_offset = body_buffer * segmentwidth
    midleft[0] -= body_offset
    xstart = midleft[0]
    ystart = midleft[1]
    segmentwidth += body_buffer
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, midleft), # get to start
        (Path.LINETO, [xstart, ystart + segmentheight/2 - curve_offset]), 
        (Path.CURVE3, [xstart, ystart + segmentheight/2]),
        (Path.LINETO, [xstart + curve_offset, ystart + segmentheight/2]),
        (Path.LINETO, [xstart + segmentwidth + body_offset, ystart + segmentheight/2]), 
        (Path.LINETO, [xstart + segmentwidth + body_offset, ystart - segmentheight/2]),
        (Path.LINETO, [xstart + curve_offset, ystart - segmentheight/2]),
        (Path.CURVE3, [xstart, ystart - segmentheight/2]),
        (Path.LINETO, [xstart, ystart - segmentheight/2 + curve_offset]),
        (Path.CLOSEPOLY, midleft) # line to beginning
        ])
    
    return codes, verts

def arcColor(magnitude, updown):
    
    '''
    Parameters
    ----------
    magnitude : floating point decimal between 0 and 1
        how dark or light do we want the shade.
    updown : string
        'up' or 'down'.

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
    
    if updown == 'up':
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
    
    body_buffer = 0.2 # fraction of segment width
    curve_buffer = 0.05 # fraction of segment height
    
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
            codes,verts = leftSegmentPatch(bod_point, body_buffer, curve_buffer, segment_height, segment_width)
        else:
            # leg_point[0] = leg_point[0] - 0.2 * leg_length
            rec = mpatches.Rectangle(leg_point, width=leg_length, height=leg_thickness, color = leg_color,
                                transform=Affine2D().rotate_deg_around(*point_of_rotation, 90-leg_angle)+ax.transData)
            codes,verts = rightSegmentPatch(bod_point, body_buffer, curve_buffer, segment_height, segment_width)
        
        
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
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.get_xaxis().set_ticks([])
    # ax.get_yaxis().set_ticks([])

    return ax

def simulate_steps(simulation):
    # leg quantity and number of cycles to show
    num_legs = simulation['num_legs']
    legs_per_side = int(num_legs/2)
    num_cycles = simulation['num_cycles']
    
    # step parameters
    gait_cycle = simulation['gait_cycle'] # in seconds
    duty_factor = simulation['duty_factor'] # in fraction of gait cycle
    opposite_offset = simulation['opposite_offset'] # in fraction of gait cycle
    anterior_offset = simulation['anterior_offset'] # in fraction of gait cycle
    fps = simulation['fps']
    
    # figure out some parameters of timing
    max_time = int(num_cycles * gait_cycle)
    frame_times = np.linspace(0,max_time,max_time*fps)
    frame_times = frame_times[1:]
    seconds_per_stance = duty_factor * gait_cycle
    opposite_offset_seconds = opposite_offset * gait_cycle
    anterior_offset_seconds = anterior_offset * gait_cycle
    
    # figure out some relatinoships between legs
    left_legs = np.array(['L' + str(n+1) for n in range(legs_per_side)])
    right_legs = np.array([x.replace('L','R') for x in left_legs])
    all_leg_list = np.hstack((left_legs, right_legs))
    anterior_left = np.roll(left_legs,1)
    anterior_right = np.roll(right_legs,1)
    anterior_leg_list = np.hstack((anterior_left,anterior_right))
    anterior_legs = dict(zip(all_leg_list, anterior_leg_list))
    opposite_legs = {}
    for i, leg in enumerate(left_legs):
        opposite_legs[leg] = right_legs[i]
    for i, leg in enumerate(right_legs):
        opposite_legs[leg] = left_legs[i]
    
    leg_ups = {}
    leg_downs = {}
    
    # set ups and downs times for rear left leg
    rear_left  = 'L' + str(legs_per_side)
    leg_ups[rear_left] = []
    leg_downs[rear_left] = [-gait_cycle, 0]
    for x in range(num_cycles):
        leg_downs[rear_left].append(leg_downs[rear_left][-1] + gait_cycle)
    leg_ups[rear_left]  = [x + seconds_per_stance for x in leg_downs[rear_left]]
    
    # get times for other left legs 
    for leg in np.flip(left_legs)[:-1]:
        # print(leg, anterior_legs[leg])
        leg_downs[anterior_legs[leg]] = [x + anterior_offset_seconds for x in leg_downs[leg]]
        leg_ups[anterior_legs[leg]] = [x + anterior_offset_seconds for x in leg_ups[leg]]
    
    # get times for all right legs
    for leg in left_legs:
        leg_downs[opposite_legs[leg]] = [x + opposite_offset_seconds for x in leg_downs[leg]]
        leg_ups[opposite_legs[leg]] = [x + opposite_offset_seconds for x in leg_ups[leg]]
        
    # trim to times we want . . .
    for leg in leg_downs.keys():
        leg_downs[leg] = [x for x in leg_downs[leg] if x >= 0]
        leg_downs[leg] = [x for x in leg_downs[leg] if x <= max_time]
        leg_ups[leg] = [x for x in leg_ups[leg] if x >= 0]
        leg_ups[leg] = [x for x in leg_ups[leg] if x <= max_time]
        
    # convert to the legacy dictionary up_down_times
    up_down_times = {}
    for leg in leg_downs.keys():
        up_down_times[leg] = {}
        up_down_times[leg]['d'] = leg_downs[leg]
        up_down_times[leg]['u'] = leg_ups[leg]
    
    return up_down_times, frame_times

if __name__ == '__main__':
    main()




