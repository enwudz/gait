# -*- coding: utf-8 -*-
"""

Make a movie showing walking patterns
from a mov_data.txt file from frame_stepper.py

To make a movie from saved frames:
ffmpeg -f image2 -r 10 -pattern_type glob -i 'boxstepper*.png' -pix_fmt yuv420p -crf 20 10fps_lateral_boxgrid_movie.mp4
ffmpeg -f image2 -r 10 -pattern_type glob -i 'gaitstyles_*.png' -pix_fmt yuv420p -crf 20 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" 10fps_gaitstyles_movie.mp4
-r is the framerate for the output movie

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
path_to_gait = '/Users/iwoods/Documents/GitHub/gait'
sys.path.insert(0, path_to_gait)
import gait_analysis
import cv2
# import matplotlib as mpl

def main(movie_folder):
    
    # ==> GET PLOT PARAMETERS   
    boxsize, stepregion, box_color, down_color, up_color = getPlotParameters()
  
    # get an array of all legs on the critter
    num_legs = 8
    up = get_legarray('up', num_legs)
    legs = np.array(sorted(np.ravel(up)))
    
    # get plot coordinates for each leg
    leg_coordinates, stepbox_size = get_legcoordinates(up, boxsize, stepregion)
        
    # ==> SAMPLE plot for chosen legs
    # legs_to_plot = np.ravel(get_legarray('first_pair',8)) 
    # plot_sample_legs(legs_to_plot, up, boxsize, stepregion, up_color, down_color, box_color)
    
    # ==> LOAD MOVIE DATA and get leg matrix for each frame
    # leg matrix = 1's are swings, 0's are stances
    # each column is a frame from a movie
    all_leg_movement_matrix = get_legmatrix(legs, movie_folder)

    # ==> GET box graphic for each frame
    # mpl.rcParams['savefig.pad_inches'] = 0
    legs_to_show = ['L1','L2','L3','R1','R2','R3']
    print('... saving frames for box movie ...')
    for frame in np.arange(np.shape(all_leg_movement_matrix)[1]):
        
        # which legs are swinging (1) vs. standing (0) in this frame
        frame_data = all_leg_movement_matrix[:,frame]
        # print(frame_data)
        
        f, a = plot_frame(frame_data, legs, up, legs_to_show, leg_coordinates, boxsize, stepbox_size, box_color, down_color, up_color)
        # plt.show()
        plt.savefig('boxstepper_' + str(frame).zfill(8) + '.png', facecolor = box_color)
        plt.close()
    
    # ==> GET GAIT STYLES for each frame
    lateral_legs = np.array(['L1','R1','L2','R2','L3','R3'])
    # print(lateral_legs)
    # lateral_leg_movement_matrix = get_legmatrix(lateral_legs, movie_folder)
    # o = open('gaits_for_frames.txt','w')
    # frame_number = 0
    # for frame in np.arange(np.shape(lateral_leg_movement_matrix)[1]):
    #     frame_number += 1
    #     frame_data = lateral_leg_movement_matrix[:,frame]
    #     gait_style = get_hexapodGaitStyle(frame_data, lateral_legs)
    #     o.write(str(frame_number) + ',' + gait_style + '\n')
    # o.close()
    
    # ==> Save blank frames with gait style labels
    # movie_file = '2a058-062_cropped.mp4'
    # vid = cv2.VideoCapture(movie_file)
    # ret, frame = vid.read()
    # fname = movie_file.split('.')[0] + '_first_frame.png'
    # cv2.imwrite(fname, frame)
    # # numframes = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    # dpi = 72
    # frame_width = vid.get(3)/dpi
    # frame_height = vid.get(4)/dpi
    # # print(frame_width, frame_height)
    # vid.release()
    
    # with open('gaits_for_frames.txt','r') as f:
    #     gait_styles_for_frames = [x.rstrip().split(',')[1] for x in f.readlines()]
    # # print(gait_styles_for_frames)
    # # print(len(gait_styles_for_frames), numframes)
    # all_combos, combo_colors = gait_analysis.get_gait_combo_colors('lateral')
    # frame_num=0
    # for gait_style in gait_styles_for_frames:

    #     frame_num += 1
    #     f,a = plt.subplots(figsize=(frame_width,frame_height), facecolor = 'k')
    #     # a = f.add_axes((1, 1, frame_width, frame_height))

    #     a.text(0.2, int(frame_height)/2, gait_style.replace('_','\n'), 
    #            fontsize=40, color = combo_colors[gait_style])
    #     a.set_facecolor('k')
    #     a.set_xlim([0,frame_width])
    #     a.set_ylim([0,frame_height])
    #     a.axis('off')
    #     # plt.show()
    #     plt.savefig('gaitstyles_' + str(frame_num).zfill(8) + '.png', facecolor = box_color)
    #     plt.close()

def get_hexapodGaitStyle(frame_data, legs):
    
    '''
    
    Gait styles for different combinations of swinging legs for hexapods (6 legs)
    
    Parameters
    ----------
    frame_data : numpy array
        1s (for swings) and 0s (for stances) for a frame of a movie
    legs : numpy array
            all of the legs that are shown in frame_data ... in order

    Returns
    -------
    gait_style : string
        'tripod_canonical' or 'tripod_other' or
        'tetrapod_canonical' or 'tetrapod_gallop' or 'tetrapod_other' or
        'pentapod' or 'stand' or 'other'

    '''
    
    # which legs are swinging?
    swinging_legs = legs[np.where(frame_data == 1)]
    
    # make the swing_combo from these swinging legs
    swing_combo = '_'.join(sorted(swinging_legs))

    # find the gait_style for this swing_combo
    gait_style = gait_analysis.get_swing_categories(swing_combo, 'lateral')

    return(gait_style)

def plot_frame(frame_data, legs, leg_array, legs_to_show, leg_coordinates, boxsize, stepbox_size, box_color, down_color, up_color):
    '''
    Parameters
    ----------
    frame_data : numpy array
        1s (for swings) and 0s (for stances) for a frame of a movie
    legs : list or numpy array
        all of the legs that are shown in frame_data ... in order
    leg_array : numpy array
        a matrix (typically 2 columns x N rows) of legs ... N = number of pairs
    legs_to_show : list or numpy array
        which legs to show in the plot.
    leg_coordinates : dictionary
        lower left coordinates of step regions for each leg
    boxsize : integer
        size of field for each leg
    stepbox_size : integer
        size of field (width and height) devoted to each step or stance.
    box_color : matplotlib color
        background color of figure.
    down_color : matplotlib color
        color to show stance (aka foot down).
    up_color : matplotlib color
        color to show swing.

    Returns
    -------
    f, a = handles for the figure and the axis

    '''
    
    # set up a figure
    fig_height, fig_width = 2*np.array(np.shape(leg_array)) 
    f = plt.figure(figsize = (fig_width, fig_height), 
                       facecolor = box_color)#  , frameon=False)
    a = f.add_axes((0, 0, 1, 1))
    
    # which legs are swinging?
    swinging_legs = legs[np.where(frame_data == 1)]
    # print(swinging_legs)
    
    # set leg colors for each leg to plot
    for leg in legs_to_show:
        x,y = leg_coordinates[leg]
        if leg in swinging_legs:
            # plot a SWING box for this leg
            rect_color = up_color        
        else:
            # plot a STANCE box for this leg
            rect_color = down_color
        rect = patches.Rectangle((x, y), stepbox_size, stepbox_size, facecolor=rect_color)
        a.add_patch(rect)
        
    a.set_facecolor(box_color)
    a.set_xlim([0,boxsize*np.shape(leg_array)[1]])
    a.set_ylim([0,boxsize*np.shape(leg_array)[0]])
    # a.set_xticks([])
    # a.set_yticks([])
    # a.get_xaxis().set_visible(False)
    # a.get_yaxis().set_visible(False)
    
    for side in ['bottom','top','right','left']:
        a.spines[side].set_color('w')
        a.spines[side].set_linewidth(3)

    # plt.autoscale(tight=True)
    
    return f, a

def getPlotParameters():
    # The size of each box (in pixels)
    boxsize = 100
    # The region of box that is devoted to showing the steps (a percentage)
    stepregion = 0.95
    # Colors for outline (box), for leg DOWN, and for leg UP
    box_color = 'k' # 'slategray'
    down_color = 'steelblue'
    up_color = 'aliceblue'
    
    return boxsize, stepregion, box_color, down_color, up_color 
    
def get_legmatrix(legs, movie_folder):    
    # get step times for each leg for which we have data
    mov_data = os.path.join(movie_folder, 'mov_data.txt')
    up_down_times, latest_event = gait_analysis.getUpDownTimes(mov_data) 
    
    # quality control on up_down_times
    gait_analysis.qcUpDownTimes(up_down_times)
    
    # Get all frame times for this movie
    frame_times = gait_analysis.get_frame_times(movie_folder)

    # trim frame_times to only include frames up to last recorded event
    last_event_frame = np.min(np.where(frame_times > latest_event*1000))
    frame_times_with_events = frame_times[:last_event_frame]
    
    # get leg matrix
    leg_matrix = gait_analysis.make_leg_matrix(legs, up_down_times, frame_times_with_events)
    return leg_matrix

def plot_sample_legs(legs_to_plot, leg_array, boxsize, stepregion, up_color, down_color, box_color):
    
    # get a dictionary of sample states (alternating up and down)
    leg_states = get_legstates(np.ravel(leg_array))
    
    # set up coordinates for each leg
    leg_coordinates, stepbox_size = get_legcoordinates(leg_array, boxsize, stepregion)
    
    fig_height, fig_width = 2*np.array(np.shape(leg_array))
    
    f,a = plt.subplots(1,1, figsize = (fig_width, fig_height) )
    
    for leg in legs_to_plot:
        if leg_states[leg] == 1:
            rect_color = up_color
        else:
            rect_color = down_color
            
        x,y = leg_coordinates[leg]
    
        rect = patches.Rectangle((x, y), stepbox_size, stepbox_size, facecolor=rect_color)
        a.add_patch(rect)
    
    a.set_facecolor(box_color)
    # a.set_xlim([0,boxsize*np.shape(leg_array)[1]])
    # a.set_ylim([0,boxsize*np.shape(leg_array)[0]])
    a.set_xticks([])
    a.set_yticks([])
    plt.show()

def get_legcoordinates(leg_array, boxsize = 100, stepregion = 0.9):
    
    box_xs = boxsize * np.arange(0, np.shape(leg_array)[1])
    box_ys = np.flip(boxsize * np.arange(0, np.shape(leg_array)[0]))
    
    box_buffer =  int ( ( boxsize - (boxsize * stepregion) ) / 2  )
    stepbox_size = int(stepregion * boxsize)
    
    stepbox_xs = box_xs + box_buffer
    stepbox_ys = box_ys + box_buffer
    
    # make dictionary of leg => coordinates
    leg_coordinates = {}
    for y in np.arange(np.shape(leg_array)[0]):
        for x in np.arange(np.shape(leg_array)[1]):
            leg = leg_array[y,x]
            xcoord = stepbox_xs[x]
            ycoord = stepbox_ys[y]
            leg_coordinates[leg] = np.array([xcoord, ycoord])
    return leg_coordinates, stepbox_size

def get_legstates(legs):
    
    even_add = np.array([1,0])
    odd_add = np.array([0,1])
    
    states = np.array([])
    
    if len(legs) % 2 != 0:
        exit('please specify an even number of legs')
    else:
        num_leg_pairs = len(legs) / 2
        
    for i in np.arange(num_leg_pairs):
        if i % 2 == 0:
            states = np.append(states, even_add)
        else:
            states = np.append(states, odd_add)

    leg_states = dict(zip(legs,states))
    
    return leg_states
    
def get_legarray(leg_group, num_legs):
    # define legs and different orientations
    legs = np.array(['L1','R1','L2','R2','L3','R3','L4','R4'])
    
    up = legs.reshape(int(num_legs/2),2)
    
    if leg_group == 'up':
        legarray = up
    elif leg_group == 'left':
        legarray = np.rot90(up)
    elif leg_group == 'down':
        legarray = np.rot90(up,2)
    elif leg_group == 'right':
        legarray = np.rot90(up,3)
    elif leg_group == 'lateral':
        legarray = np.ravel(up[:-1,:])
    elif leg_group == 'left_lateral':
        legarray = up[:-1,0]
    elif leg_group == 'right_lateral':
        legarray = up[:-1,1]
    elif leg_group == 'first_pair':
        legarray = up[0,:]
    elif leg_group == 'second_pair':
        legarray = up[1,:]
    elif leg_group == 'third_pair':
        legarray = up[2,:]
    elif leg_group == 'fourth_pair':
        legarray = up[3,:]
    else:
        legarray = up
    return legarray
    
    
if __name__ == "__main__":

    if len(sys.argv) > 1:
        movie_folder = sys.argv[1]
    else:
        dirs = next(os.walk(os.getcwd()))[1]
        dirs = sorted([d for d in dirs if d.startswith('_') == False and d.startswith('.') == False])
        movie_folder = dirs[0]
        
    print('Getting data from ' + movie_folder)

    main(movie_folder)