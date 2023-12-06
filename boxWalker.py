#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Make frames for an animation showing gait styles at each frame
    from gait_styles sheet of excel file (from frameStepper)

to make a movie from saved frames, run:
    python makeMovieFromImages.py '*searchterm*' fps outfile

WishList:
    set text position based on leg coordinates
    
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
import gaitFunctions
import pandas as pd

def main(movie_file, leg_set = 'lateral'):
    
    ## Figure out what species we are dealing with
    identity_info = gaitFunctions.loadIdentityInfo(movie_file)

    if 'species' in identity_info.keys():
        species = identity_info['species']
        leg_group = 'up' # choose orientation here
    else:
        species = 'tardigrade'
        leg_group = 'up'
    if 'num_legs' in identity_info.keys():
        num_legs = identity_info['num_legs']
    else:
        num_legs = 8
    print('\nThis is a ' + species + ' with ' + str(num_legs) + ' legs to show')
    print('we will show it going ' + leg_group)
    
    # get an array showing the orientation of the legs in the plot
    if species == 'tardigrade':
        leg_set = gaitFunctions.choose_lateral_rear()
        leg_array = get_legarray(leg_group, num_legs)
    else:
        leg_set = species
        leg_array = get_legarray(leg_group, num_legs + 2)
    
    # ==> GET PLOT PARAMETERS   
    boxsize, stepregion, box_color, down_color, up_color = getPlotParameters()
  
    # get dictionary of leg_name => lower left box coordinates and size of box showing steps
    leg_coordinates, stepbox_size = get_legcoordinates(leg_array, boxsize, stepregion) 
    # print(leg_coordinates)
            
    # ==> SAMPLE plot for chosen legs
    # legs_to_plot = np.ravel(get_legarray('all', num_legs)) 
    # plot_sample_legs(legs_to_plot, leg_array, boxsize, stepregion, up_color, box_color)
    
    # ==> Load gait style data
    filestem = movie_file.split('.')[0]
    excel_file = filestem + '.xlsx'
    gait_data = pd.read_excel(excel_file, sheet_name='gait_styles')
    times = gait_data.frametimes.values
    
    if leg_set == 'rear':
   
        gait_styles = gait_data['gaits_rear'].values
        swings = gait_data['swinging_rear'].values
        combos, combo_colors = gaitFunctions.get_gait_combo_colors('rear')
        legs_to_show = gaitFunctions.get_leg_combos()[0]['rear']
        text_x, text_y = (70,140)
        
    elif leg_set == 'lateral':
        
        gait_styles = gait_data['gaits_lateral'].values
        swings = gait_data['swinging_lateral'].values
        combos, combo_colors = gaitFunctions.get_gait_combo_colors('lateral')
        legs_to_show = gaitFunctions.get_leg_combos()[0]['lateral']
        text_x, text_y = (45,40)
        
    else:
        
        gait_styles = gait_data['gaits'].values
        swings = gait_data['swinging_leg'].values
        combos, combo_colors = gaitFunctions.get_gait_combo_colors(num_legs)
        legs_to_show = gaitFunctions.get_leg_list(4)
        text_x, text_y = (15,95) # this is assuming rightward orientation

    # ==> Save image for each frame
    boxwalker_folder = filestem + '_boxwalker_' + leg_set
    os.mkdir(boxwalker_folder)
    
    print('... saving frames for box movie in ' + boxwalker_folder + ' ...')
    
    fig_height, fig_width = 2*np.array(np.shape(leg_array))
    # print('height',fig_height)
    # print('width',fig_width)
    
    for i, frame_time in enumerate(times):
        try: # if no swinging legs, then we have nan, which we cannot split
            swinging_legs = swings[i].split('_')
        except:
            swinging_legs = []
        gait_style = gait_styles[i]
        gait_color = combo_colors[gait_style]

        f = plt.figure(figsize = (fig_width, fig_height), 
                           facecolor = box_color)#  , frameon=False)
        a = f.add_axes((0, 0, 1, 1))
        
        # set leg colors for each leg to plot
        for leg in legs_to_show:
            x,y = leg_coordinates[leg]
            if leg in swinging_legs:
                # plot a SWING box for this leg
                # rect_color = up_color        
                rect_color = gait_color
            else:
                # plot a STANCE box for this leg
                rect_color = down_color
            
            rect = patches.Rectangle((x, y), stepbox_size, stepbox_size, facecolor=rect_color)
            a.add_patch(rect)
            
        a.set_facecolor(box_color)
        a.set_xlim([0,boxsize*np.shape(leg_array)[1]])
        a.set_ylim([0,boxsize*np.shape(leg_array)[0]])
        
        # set bounding box
        for side in ['bottom','top','right','left']:
            a.spines[side].set_color('k')
            a.spines[side].set_linewidth(3)
            
        # add text for gait style ... if tardigrade!
        # if species == 'tardigrade':
        a.text(text_x, text_y, s=gait_style.replace('_','\n'), color=gait_color, fontsize=30, fontweight='bold')
        
        # plt.show()
        fname = os.path.join(boxwalker_folder, filestem + '_boxwalker_' + str(int(frame_time*1000)).zfill(6)  + '.png')
        # print(fname)
        plt.savefig(fname, facecolor = box_color)
        plt.close()

def getPlotParameters():
    # The size of each box (in pixels)
    boxsize = 100
    # The region of box that is devoted to showing the steps (a percentage)
    stepregion = 0.95
    # Colors for outline (box), for leg DOWN, and for leg UP
    box_color = 'k' # 'slategray'
    
    # get standard stance and swing colors
    # down_color, up_color = gaitFunctions.stanceSwingColors()
    
    # OR choose stance and swing colors
    down_color = 'dimgray' # 'steelblue'
    up_color = 'aliceblue' 
    
    return boxsize, stepregion, box_color, down_color, up_color   

def plot_sample_legs(legs_to_plot, leg_array, boxsize, stepregion, up_color, box_color):
    
    print('Test plot of ', legs_to_plot)
    
    # set up coordinates for each leg
    leg_coordinates, stepbox_size = get_legcoordinates(leg_array, boxsize, stepregion)
    
    fig_height, fig_width = 2*np.array(np.shape(leg_array))
    
    f,a = plt.subplots(1,1, figsize = (fig_width, fig_height) )
    
    for leg in legs_to_plot:
            
        x,y = leg_coordinates[leg]
        print(leg,x,y)
    
        rect = patches.Rectangle((x, y), stepbox_size, stepbox_size, facecolor=up_color)
        a.add_patch(rect)
    
    a.set_facecolor(box_color)
    a.set_xlim([0,boxsize*np.shape(leg_array)[1]])
    a.set_ylim([0,boxsize*np.shape(leg_array)[0]])
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
    
def get_legarray(leg_group, num_legs):
    
    # define legs and different orientations
    legs = np.array(gaitFunctions.get_leg_list(num_legs))
    
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
    
if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])

    main(movie_file)