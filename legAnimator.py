#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 21:58:21 2023

@author: iwoods

animation of legs

(option to) add animated step plot along with critter?

crop area of movie: https://ezgif.com/crop-video
make a gif: https://ezgif.com/video-to-gif

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import gaitFunctions
from matplotlib.animation import FuncAnimation


def main():
    
    animation_fps = 30
    critter = 'tardigrade'
    save_animation = True # true or false
    make_excel = True # true or false
    
    ## ==> get SIMULATED data based on step parameters
    
    num_legs = gaitFunctions.getFeetFromSpecies(critter)
    up_down_times, frame_times = load_simulated_steps(num_legs)

    ## ==> OR get MEASURED up / down times for legs from an experiment
    # up_down_times, frame_times = load_movie_steps()
    
    # print(up_down_times)
    
    ## ==> make 2 dictionaries:
    ## legstates = a dictionary of leg states (up or down)
    ##     keys = leg names, values = list of up or downs
    ## legangles = a dictionary of swing extents (backward = 0, forward = 1)
    ##     keys = leg names, values = array of swing extents
    legstates, legangles = get_leg_swings(up_down_times, frame_times)

    ## ==> make basic animation
    fname = basic_animation(legangles, legstates, critter, animation_fps, save_animation) # True to save the animation      

    ## ==> make excel file from simulated steps
    if len(fname) > 0 and make_excel:
        makeExcelFromSimulatedSteps(fname, critter, up_down_times)

def load_simulated_steps(num_legs):
    ## define step parameters
    num_cycles = 2
    gait_cycle = 2 # in seconds
    duty_factor =     0.67 # in fraction of gait cycle
    anterior_offset = 0.33 # in fraction of gait cycle
    opposite_offset = 0.5 # in fraction of gait cycle
    fps = 30 # frames per second
    
    simulation = {}
    simulation['num_legs'] = num_legs
    simulation['num_cycles'] = num_cycles
    simulation['gait_cycle'] = gait_cycle # in seconds
    simulation['duty_factor'] = duty_factor
    simulation['opposite_offset'] = opposite_offset
    simulation['anterior_offset'] = anterior_offset
    simulation['fps'] = fps
    
    max_time = int(num_cycles * gait_cycle)
    frame_times = np.linspace(0,max_time,max_time*fps)
    simulation['frame_times'] = frame_times[1:]
    simulation['max_time'] = max_time
    
    up_down_times, frame_times = simulate_steps(simulation)
    return up_down_times, frame_times

def makeExcelFromSimulatedSteps(fname, critter, up_down_times):
    
    # make identity tab
    from datetime import date 
    import calendar
    import pandas as pd
    info = {}
    todays_date = date.today() 
    fstem = fname.split('.')[0]
    excel_filename = fstem + '.xlsx'
 
    info['file_stem'] = fname.split('.')[0]
    info['month'] = calendar.month_abbr[todays_date.month].lower()
    info['date'] = todays_date.day
    info['species'] = critter
    species = ['human','tardigrade','cat','dog','insect','tetrapod','hexapod']
    num_legs = [2, 8, 4, 4, 6, 4, 6]
    species_legs = dict(zip(species,num_legs))
    info['num_legs'] = 6 # guessing
    for thing in species:
        if thing in critter:
            info['num_legs'] = species_legs[thing]
    info['treatment'] = 'simulation'
    info['individualID'] = 1
    info['initials'] = 'ani'
    info['width'], info['height'], info['fps'], info['#frames'], info['duration'] = gaitFunctions.getVideoData(fname, False)
    info['time_range'] ='0-' + str(info['duration'])

    print_order = gaitFunctions.identity_print_order()
    vals = [info[x] for x in print_order if x in print_order]
    d = {'Parameter':print_order,'Value':vals}
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename) as writer:
        df.to_excel(writer, index=False, sheet_name='identity')
            
    # make pathtracking tab
    pathtracking = {}
    frame_times = gaitFunctions.getFrameTimes(fname)
    pathtracking['times'] = frame_times
    pathtracking['xcoords'] = np.linspace(0,900,len(frame_times))
    pathtracking['ycoords'] = 100 * np.ones(len(frame_times))
    pathtracking['areas'] = 1000 * np.ones(len(frame_times))
    pathtracking['lengths'] = 100 * np.ones(len(frame_times))
    pathtracking['tracking_uncertainty_25'] = np.array(['FALSE'] * len(frame_times))
    path_df = pd.DataFrame(pathtracking)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        path_df.to_excel(writer, index=False, sheet_name='pathtracking')   
    
    # make path_stats tab
    import analyzeTrack
    analyzeTrack.main(fname)
    
    # make steptracking tab
    
    # run analyzeSteps to make remaining tabs
    
    
    
    
    

def basic_animation(legangles, legstates, critter, fps=30, save_animation=False):    
    
    frame_interval = int((1 / fps) * 1000)
    
    fig, ax = plt.subplots(figsize=(7,8))
    ani = FuncAnimation(fig, animate_steps, frames=len(legangles['L1']), 
                        interval=frame_interval, repeat=False, fargs=[ax, legangles, legstates, critter]) 
    
    if save_animation:
        fname = 'ani_' + critter + '_' + str(fps) + '.mp4'
        ani.save(fname)
    else:
        fname = ''
    plt.show()
    return fname

def animate_steps(i, ax, legangles, legstates, critter = 'tardigrade'):
        
    legs = list(legangles.keys())
    
    ax.clear()
    
    legstate = {}
    swingextent = {}
    
    for leg in legs:
        legstate[leg] = legstates[leg][i]
        swingextent[leg] = legangles[leg][i]
    
    ax = drawLegs(ax, critter, swingextent, legstate)
    
    num_segments = int(len(legs)/2)
    
    # add head
    if critter in ['tardigrade','cat']:
        ax = add_head(ax, num_segments, critter)

    # add tail
    if critter in ['tardigrade','cat']:
        ax = add_tail(ax, num_segments, critter)
        
    return

def add_tail(ax, num_segments, critter):
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    segmentwidth = (xlims[1]-xlims[0]) / 2
    segmentheight = (xlims[1]-xlims[0]) / num_segments
    
    bottom_middle = [ xlims[0] + segmentwidth , ylims[0] ]
    
    if critter == 'cat':
        
        codes,verts = cat_tail(bottom_middle, segmentwidth, segmentheight)
        tail = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
        ax.add_patch(tail)
        ax.set_ylim([ylims[0] - (1.6*segmentheight), ylims[1] ])
        
    else:
        
        codes,verts = tardigrade_tail(bottom_middle, segmentwidth, segmentheight)
        tail = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
        ax.add_patch(tail)
        ax.set_ylim([ylims[0] - (0.5*segmentheight), ylims[1] ])
        
    return ax

def add_head(ax, num_segments, critter = 'tardigrade'):
    
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    
    segmentwidth = (xlims[1]-xlims[0]) / 2
    segmentheight = (xlims[1]-xlims[0]) / num_segments
    
    top_middle = [ xlims[0] + segmentwidth , ylims[1] ]
    
    if critter == 'cat':
        codes,verts = cat_head(top_middle, segmentwidth, segmentheight)
        head = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
        ax.add_patch(head)
    else:
        headcodes, headverts, lefteye, righteye , ls_codes, ls_verts, rs_codes, rs_verts = tardigrade_head(top_middle,segmentwidth, segmentheight)
        head = mpatches.PathPatch(mpath.Path(headverts, headcodes), fc='k')
        ax.add_patch(head)
        ax.add_patch(lefteye)
        ax.add_patch(righteye)
        lstylet = mpatches.PathPatch(mpath.Path(ls_verts, ls_codes), ec='w', lw=3, fc = 'none')
        ax.add_patch(lstylet)
        rstylet = mpatches.PathPatch(mpath.Path(rs_verts, rs_codes), ec='w', lw=3, fc = 'none')
        ax.add_patch(rstylet)
       
    ax.set_ylim([ylims[0], ylims[1]+(1.1*segmentheight)])
    
    return ax
    

def get_leg_swings(up_down_times, frame_times):    
    ## get legs, with specified leg order
    num_legs = len(up_down_times)
    legs = gaitFunctions.get_leg_list(num_legs)
    legs = [x for x in legs if x in up_down_times.keys()]
    
    # get leg matrix
    leg_matrix = np.zeros([len(legs), len(frame_times)])
    leg_matrix = gaitFunctions.fill_leg_matrix(leg_matrix, legs, up_down_times, frame_times)
    
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

def rightSegmentPatch(startpoint, curve_buffer, segmentheight, segmentwidth):

    curve_offset = curve_buffer * segmentheight
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth - curve_offset, startpoint[1] + 0.5 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] + 1 * segmentwidth, startpoint[1] + 0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth, startpoint[1] + 0.5 * segmentheight - curve_offset ]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth, startpoint[1] - 0.5 * segmentheight + curve_offset ]),
        (Path.CURVE3, [startpoint[0] + 1 * segmentwidth, startpoint[1] - 0.5 * segmentheight]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth - curve_offset, startpoint[1] - 0.5 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 0 * segmentwidth, startpoint[1] - 0.5 * segmentheight]),
        (Path.CLOSEPOLY, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        ])
    
    return codes, verts

def leftSegmentPatch(startpoint, curve_buffer, segmentheight, segmentwidth):

    curve_offset = curve_buffer * segmentheight
    Path = mpath.Path
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 1 * segmentwidth + curve_offset, startpoint[1] + 0.5 * segmentheight ]),
        (Path.CURVE3, [startpoint[0] - 1 * segmentwidth, startpoint[1] + 0.5 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 1 * segmentwidth, startpoint[1] + 0.5 * segmentheight - curve_offset ]),
        (Path.LINETO, [startpoint[0] - 1 * segmentwidth, startpoint[1] - 0.5 * segmentheight + curve_offset ]),
        (Path.CURVE3, [startpoint[0] - 1 * segmentwidth, startpoint[1] - 0.5 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 1 * segmentwidth + curve_offset, startpoint[1] - 0.5 * segmentheight]),
        (Path.LINETO, [startpoint[0] - 0 * segmentwidth, startpoint[1] - 0.5 * segmentheight]),
        (Path.CLOSEPOLY, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
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

def drawLegs(ax, critter, swingextents, legstates):
    '''
    
    Parameters
    ----------
    ax : matplotlib axis object
    critter : string
        species
    swingextents : dictionary
        keys = names of legs
        values = floating point decimal between 0 and 1
        1 = anterior-most position of leg, 0 = posterior-most position of leg
    legstates : dictionary
        keys = names of legs    
        values = state of leg 'up' for swing, 'down' for stance.

    Returns
    -------
    ax : matplotlib axis object with all legs drawn in appropriate positions

    '''
    
    # how thick and long should we make the legs?
    leg_thickness = 0.6 # height
    leg_length = 1.6 # width
    
    # how wide and tall should we make the segments?
    segment_width = 0.8 * leg_length
    segment_height = 1.8 * leg_length
    
    # how much of the body should cover the legs, and how much curve in each segment?
    body_buffer = 0.2 * segment_width # fraction of segment width
    curve_buffer = 0.05 # fraction of segment height
    
    # get a list of legs that we need to worry about here
    all_legs = gaitFunctions.get_leg_list(10)
    legs = [x for x in all_legs if x in swingextents.keys()]

    # set up number of body segments, depending on number of legs
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
    body_ys = {}
    start_x = 0
    start_y = 0
    counter = 0
    
    for row in range(num_rows):
        # row =  each segment
        
        # col = left or right. Left leg = col 0,2,4 ...; right leg = col 1,3,5 ...
        for col in range(num_cols):
            
            # find starting x point for this leg
            if col % 2 == 0: # a left leg!
                this_x = start_x - segment_width
            else: # a right leg!
                this_x = start_x + segment_width

            # add row offset to starting y point
            this_y = start_y - row * segment_height
            body_ys[legs[counter]] = this_y
            
            # get point for this leg
            leg_points[legs[counter]] = [this_x, this_y - leg_thickness/2]

            counter += 1
    
    # go through legs and plot
    for leg in legs:
    
        # get color and angle for this leg
        leg_color = arcColor(swingextents[leg], legstates[leg])
        leg_angle = swingAngle(swingextents[leg])
        leg_point = leg_points[leg]
        bod_point = [0, body_ys[leg]]
        
        # draw the leg and the body segment
        if 'L' in leg: # a left leg!
            leg_point = [leg_point[0] + body_buffer, leg_point[1]]
            point_of_rotation = np.array([leg_point[0], leg_point[1] + leg_thickness/2])
            rec = mpatches.Rectangle(leg_point, width=leg_length, height=leg_thickness, color = leg_color,
                                transform=Affine2D().rotate_deg_around(*point_of_rotation, 90+leg_angle)+ax.transData)
            if critter != 'human':                    
                codes,verts = leftSegmentPatch(bod_point, curve_buffer, segment_height, segment_width)
        else: # a right leg!
            leg_point = [leg_point[0] - body_buffer, leg_point[1]]
            point_of_rotation = np.array([leg_point[0], leg_point[1] + leg_thickness/2])
            rec = mpatches.Rectangle(leg_point, width=leg_length, height=leg_thickness, color = leg_color,
                                transform=Affine2D().rotate_deg_around(*point_of_rotation, 90-leg_angle)+ax.transData)
            if critter != 'human':
                codes,verts = rightSegmentPatch(bod_point, curve_buffer, segment_height, segment_width)

        # plt.scatter(point_of_rotation[0],point_of_rotation[1], c='r', s=50) # check leg point
        ax.add_patch(rec)

        if critter == 'human':
            bod_point = [bod_point[0], bod_point[1] - 0.5 * segment_height]
            codes, verts, larm_codes, larm_verts, rarm_codes, rarm_verts, head_codes, head_verts = human_torso(bod_point, segment_width, segment_height)

            bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
            ax.add_patch(bod)

            larm = mpatches.PathPatch(mpath.Path(larm_verts, larm_codes), ec='w', lw=3, fc = 'none')
            ax.add_patch(larm)

            rarm = mpatches.PathPatch(mpath.Path(rarm_verts, rarm_codes), ec='w', lw=3, fc = 'none')
            ax.add_patch(rarm)

            head = mpatches.PathPatch(mpath.Path(head_verts, head_codes), ec='w', lw=3, fc = 'none')
            ax.add_patch(head)

        else:
            bod = mpatches.PathPatch(mpath.Path(verts, codes), fc='k')
            ax.add_patch(bod)
    
    # set axis limits
    ax.set_aspect('equal')
    ax.set_xlim([-1.1* body_width/2, 1.1 * body_width/2])
    ax.set_ylim([ segment_height/2 - body_length, segment_height/2 ] )
    
    # set background color
    ax.set_facecolor("steelblue") # slategray
    
    # # clear the frame and ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    return ax

def human_torso(startpoint, segmentwidth, segmentheight):
    Path = mpath.Path
    segmentwidth = 1.5 * segmentwidth

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

def tardigrade_tail(startpoint, segmentwidth, segmentheight):

    segmentwidth = 0.38 * segmentwidth # how much of the axis do we want to fill with head?
    
    Path = mpath.Path
    
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        (Path.LINETO, [startpoint[0] -1 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.45 * segmentwidth, startpoint[1] - 0.4 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.45 * segmentwidth, startpoint[1] - 0.4 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 1 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        (Path.CLOSEPOLY, [startpoint[0] + 0 * segmentwidth, startpoint[1] + 0 * segmentheight ]),
        ])
    
    return codes, verts

def cat_tail(startpoint, segmentwidth, segmentheight):

    segmentwidth = 0.8 * segmentwidth
    segmentheight = 1 * segmentheight
    Path = mpath.Path
    
    codes, verts = zip(*[
        (Path.MOVETO, [startpoint[0] - 0.05 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.1 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.3 * segmentwidth, startpoint[1] - 0.56 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.1 * segmentwidth, startpoint[1] - 1.1 * segmentheight ]),
        (Path.LINETO, [startpoint[0] - 0.15 * segmentwidth, startpoint[1] - 1.3 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.15 * segmentwidth, startpoint[1] - 1.4 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] - 0.05 * segmentwidth, startpoint[1] - 1.45 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0 * segmentwidth, startpoint[1] - 1.35 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0.15 * segmentwidth, startpoint[1] - 1.1 * segmentheight ]),
        (Path.CURVE4, [startpoint[0] + 0 * segmentwidth, startpoint[1] - 0.56 * segmentheight ]),
        (Path.LINETO, [startpoint[0] + 0.15 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        (Path.CLOSEPOLY, [startpoint[0] - 0.05 * segmentwidth, startpoint[1] - 0 * segmentheight ]),
        ])
    
    return codes, verts

def tardigrade_head(startpoint, segmentwidth, segmentheight):
    
    segmentwidth = 0.4 * segmentwidth # how much of the axis do we want to fill with head?
    segmentheight = 0.85 * segmentheight
    
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
    
    segmentwidth = 0.6 * segmentwidth # how much of the axis do we want to fill with head?
    
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
    max_time = simulation['max_time']
    frame_times = simulation['frame_times']
    
    # figure out some parameters of timing
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




