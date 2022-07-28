#!/usr/bin/python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import os
import glob
import shutil
import sys
import cv2
from scipy.stats import sem

def makeLegDict():
    leg_dict = {}
    legs = ['R1', 'R2', 'R3', 'R4', 'L1', 'L2', 'L3', 'L4']
    gait_data = ['stance_times', 'swing_times', 'duty_factors', 'gait_cycles']
    interleg_data = ['down_to_up_opposite', 'down_to_up_anterior',
                     'down_to_down_opposite', 'down_to_down_anterior',
                     'up_to_up_opposite', 'up_to_up_anterior']
    for leg in legs:
        leg_dict[leg] = {}
        for g in gait_data:
            leg_dict[leg][g] = []
        for i in interleg_data:
            leg_dict[leg][i] = []

    return leg_dict


def save_stance_figures(data_folder, leg_dict, legs):
    for x in ['stance', 'swing']:
        box_data, f, a = plot_stance(leg_dict, legs, x, False)
        fname = os.path.join(data_folder, x + '_plot.png')
        plt.savefig(fname)


def save_leg_figures(data_folder, leg_dict, video_end):
    leg_combos = get_leg_combos()
    for legs in leg_combos.keys():
        f, a = plot_legs(leg_dict, leg_combos[legs], video_end, False)
        figname = '_'.join(leg_combos[legs]) + '.png'
        figname_path = os.path.join(data_folder, figname)
        plt.savefig(figname_path)


# from list of leg_downs, leg_ups, get stance_times, swing_times, gait_cycles, duty_factors
def getStepSummary(downs, ups):

    # only want COMPLETE gait cycles ... so a Down first and a Down last
    if downs[0] > ups[0]:
        ups = ups[1:] # if an UP comes first, get rid of it
    if ups[-1] > downs[-1]:
        ups = ups[:-1] # if an UP comes last, get rid of it

    stance_times = []
    swing_times = []
    gait_cycles = []
    duty_factors = []
    mid_swings = []

    # skip if not enough data, or if data bad
    if len(downs) < 2:
        print('No stance/swing/gait data for this leg')
    elif len(downs) != len(ups) + 1:
        print('Mismatch steps for this leg')
    else:
        stance_times = getIntervals(downs, ups, 4) # getIntervals wants to know how much to round values
        swing_times = getIntervals(ups, downs, 4)
        gait_cycles = getIntervals(downs[:-1], downs[1:], 4)
        duty_factors = np.around(np.array(stance_times) / np.array(gait_cycles), 4)

        half_swings = np.array(swing_times) / 2
        mid_swings = np.around(ups + half_swings, 4)

    return downs, ups, stance_times, swing_times, gait_cycles, duty_factors, mid_swings


# add data from up_down_times (one video) to leg_dict (multiple videos)
def addDataToLegDict(leg_dict, up_down_times):
    opposite_dict, anterior_dict = getOppAndAntLeg()

    for leg in sorted(leg_dict.keys()):
        downs = up_down_times[leg]['d']
        ups = up_down_times[leg]['u']

        downs, ups, stance_times, swing_times, gait_cycles, duty_factors = getStepSummary(downs, ups)

        if len(stance_times) == 0:
            print('No gait data for ' + leg)

        else:
            leg_dict[leg]['stance_times'].extend(stance_times)
            leg_dict[leg]['swing_times'].extend(swing_times)
            leg_dict[leg]['gait_cycles'].extend(gait_cycles)
            leg_dict[leg]['duty_factors'].extend(duty_factors)

            # timing with other legs
            # down to down (opposite, anterior)
            # down to up (opposite, anterior)
            # up to up (opposite, anterior)
            opposite_leg = opposite_dict[leg]
            anterior_leg = anterior_dict[leg]

            ups_anterior = up_down_times[anterior_leg]['u']
            ups_opposite = up_down_times[opposite_leg]['u']

            downs_anterior = up_down_times[anterior_leg]['d']
            downs_opposite = up_down_times[opposite_leg]['d']

            # down_to_up_opposite
            down_to_up_opposite = getIntervals(downs, ups_opposite)
            leg_dict[leg]['down_to_up_opposite'].extend(down_to_up_opposite)

            # down_to_up_anterior
            down_to_up_anterior = getIntervals(downs, ups_anterior)
            leg_dict[leg]['down_to_up_anterior'].extend(down_to_up_anterior)

            # down_to_down_opposite
            down_to_down_opposite = getIntervals(downs, downs_opposite)
            leg_dict[leg]['down_to_down_opposite'].extend(down_to_down_opposite)

            # down_to_down_anterior
            down_to_down_anterior = getIntervals(downs, downs_anterior)
            leg_dict[leg]['down_to_down_anterior'].extend(down_to_down_anterior)

            # up_to_up_opposite
            up_to_up_opposite = getIntervals(ups, ups_opposite)
            leg_dict[leg]['up_to_up_opposite'].extend(up_to_up_opposite)

            # up_to_up_anterior
            up_to_up_anterior = getIntervals(ups, ups_anterior)
            leg_dict[leg]['up_to_up_anterior'].extend(up_to_up_anterior)

    return leg_dict


# getIntervals =
# a function to take two lists and return a list of intervals between the list items
# every item in list 1 will have an associated interval
# set so beginning = lowest value in list 1, and end = highest value in list 2
def getIntervals(list1, list2, dec_round=3):
    intervals = []

    if list1[-1] >= list2[-1]:
        # last item of list1 is greater than last item of list2
        # ... so we will ignore the last item of list1
        list1 = list1[:-1]

    if list2[0] <= list1[0]:
        # first item of list2 is less than first item of list1
        # ... so we will ignore the first item of list2
        list2 = list2[1:]

    if len(list2) == 0 or len(list1) == 0:
        return []

    arr1 = np.array(list1)
    arr2 = np.array(list2)

    for item in arr1:
        next_item = arr2[np.argmax(arr2 > item)]
        interval = next_item - item
        intervals.append(interval)

    return np.round(np.array(intervals), dec_round)


def get_leg_combos():
    leg_combos = {}
    leg_combos['legs_all'] = ['L4', 'L3', 'L2', 'L1', 'R1', 'R2', 'R3', 'R4']
    leg_combos['legs_lateral'] = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3']
    leg_combos['legs_all_right'] = ['R4', 'R3', 'R2', 'R1']
    leg_combos['legs_all_left'] = ['L4', 'L3', 'L2', 'L1']
    leg_combos['legs_right'] = ['R3', 'R2', 'R1']
    leg_combos['legs_left'] = ['L3', 'L2', 'L1']
    leg_combos['legs_1'] = ['R1', 'L1']
    leg_combos['legs_2'] = ['R2', 'L2']
    leg_combos['legs_3'] = ['R3', 'L3']
    leg_combos['legs_4'] = ['R4', 'L4']
    return leg_combos


# get dictionaries of opposite and anterior legs
# keyed by each leg
def getOppAndAntLeg():
    legs = ['R1', 'R2', 'R3', 'R4', 'L1', 'L2', 'L3', 'L4']
    opposites = ['L1', 'L2', 'L3', 'L4', 'R1', 'R2', 'R3', 'R4']
    anteriors = ['R3', 'R1', 'R2', 'R3', 'L3', 'L1', 'L2', 'L3']
    opposite_dict = dict(zip(legs, opposites))
    anterior_dict = dict(zip(legs, anteriors))
    return opposite_dict, anterior_dict

# get dictionary of posterior legs keyed by each leg
def getPosteriorLeg():
    legs = ['R1', 'R2', 'R3', 'R4', 'L1', 'L2', 'L3', 'L4']
    posteriors = ['R2','R3','R1','R1','L2','L3','L1','L1']
    posterior_dict = dict(zip(legs, posteriors))
    return posterior_dict

def plot_legs(legDict, legs, video_end, show=True):
    leg_yvals = list(range(len(legs)))

    # start a plot for the data
    figheight = len(legs)
    (f, a) = plt.subplots(1, figsize=(10, figheight))  # set height on # of legs

    # add a leg to the plot
    for i, leg in enumerate(legs):
        yval = leg_yvals[i]
        fd = legDict[leg]['d']
        fu = legDict[leg]['u']
        f, a = addLegToPlot(f, a, yval, fd, fu, video_end)

    # show the plot
    a.set_xlim([0, video_end])
    y_buff = 0.5
    a.set_ylim([leg_yvals[0] - y_buff, leg_yvals[-1] + y_buff])
    a.set_yticks(leg_yvals)
    a.set_yticklabels(legs)
    a.set_xlabel('Time (sec)')
    a.set_ylabel('Legs')
    plt.subplots_adjust(bottom=0.3)
    if show:
        plt.show()
    return f, a


# test if a file can be found
# input = path to file
def fileTest(fname):
    file_test = glob.glob(fname)
    if len(file_test) == 0:
        err = 'could not find ' + fname
        print(err)
        sys.exit()
        
    else:
        print('Found ' + fname)


# getUpDownTimes = a function to open a given file
# (formatted as mov_data.txt, from frame_stepper)
# and get up/down timing for a given leg
# input = path to file
# output = leg_dict, latest_datapoint
#    where leg_dict = a dictionary of lists, of up and down timing
#    keyed by leg, e.g. leg_dict['R4']['u']  ( = [ 2,5,6,8 ... ] )
def getUpDownTimes(mov_data):
    # fileTest(mov_data)

    up_down_times = {}
    latest_event = 0

    with open(mov_data, 'r') as f:
        for line in f:
            if line.startswith('Length'):
                movieLength = float(line.rstrip().split()[1])
                # do not really care about movie length
                # instead, care about the timing of the latest event recorded
            if line.startswith('Data'):
                currentLeg = line.rstrip().split()[2]
                up_down_times[currentLeg] = {}
            if line.startswith('Foot Down'):
                line = line.rstrip()
                footdown = parseFootLine(line)
                up_down_times[currentLeg]['d'] = footdown
                if max(footdown) > latest_event:
                    latest_event = max(footdown)
            if line.startswith('Foot Up'):
                line = line.rstrip()
                footup = parseFootLine(line)
                up_down_times[currentLeg]['u'] = footup
                if max(footup) > latest_event:
                    latest_event = max(footup)

    return up_down_times, latest_event

# quality control for up_down_times ... make sure up and down times are alternating!
def qcUpDownTimes(up_down_times):
    for leg in up_down_times.keys():
        downs = up_down_times[leg]['d']
        ups = up_down_times[leg]['u']
        combo_times = np.array(downs + ups)
        down_array = ['d'] * len(downs)
        up_array = ['u'] * len(ups)
        combo_ud = np.array(down_array + up_array)
        inds = combo_times.argsort()
        sorted_ud = combo_ud[inds]
        for i in range(len(sorted_ud[:-1])):
            if sorted_ud[i] == sorted_ud[i + 1]:
                print('alternating u/d problem for ' + leg)


def selectOneFromList(li):
    print('\nChoose from this list : ')
    i = 1
    li = sorted(li)
    
    for thing in li:
        print(str(i) + ': ' + thing)
        i += 1
    entry = input('\nWhich ONE do you want? ')
    choice = int(entry)
    ind = choice - 1
    print('\nYou chose ' + li[ind] + '\n')
    return li[ind]


# given a list, select a single item or multiple items or all
# return a list of items selected
def selectMultipleFromList(li):
    print('\nChoose from this list (separate by commas if multiple choices): ')
    i = 1
    for thing in li:
        print(str(i) + ': ' + thing)
        i += 1
    print(str(i) + ': select ALL')

    entry = input('\nWhich number(s) do you want? ')

    if ',' in entry:  # multiple choices selected

        indices = [int(x) - 1 for x in entry.split(',')]
        choices = [li[ind] for ind in indices]

        print('You chose: ' + ' and '.join(choices))

        return choices

    else:
        choice = int(entry)
        if choice <= len(li):
            ind = choice - 1
            print('\nYou chose ' + li[ind] + '\n')
            return [li[ind]]
        else:
            print('\nYou chose them all\n')
            return li

def pathToData(data_path):
    if '/' in data_path:
        folders = data_path.split('/')[1:]
    elif '\\' in data_path:
        folders = data_path.split('\\')[1:]
    os_path = os.path.join(os.sep, os.sep.join(folders))
    return os_path

def listDirectories():
    dirs = next(os.walk(os.getcwd()))[1]
    dirs = sorted([d for d in dirs if d.startswith('_') == False and d.startswith('.') == False])
    return dirs


def getMovieFromFileList(movie_folder): # movie_folder needs to be complete path
    
    file_list = glob.glob(os.path.join(movie_folder, '*'))

    movieList = []
    for f in file_list:
        if '.mov' in f or '.mp4' in f or '.avi' in f:
            movieList.append(f.split('/')[-1])

    if len(movieList) > 1:
        exit('I found ' + str(len(movieList)) + 'movies in ' + movie_folder)
    else:
        return (movieList[0])


def getVideoStats(vid, printout=True):
    numframes = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vidfps = vid.get(cv2.CAP_PROP_FPS)
    vidstart = vid.get(cv2.CAP_PROP_POS_MSEC)

    vidlength = round(numframes / vidfps, 3)

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    if printout is True:
        print('width = ', frame_width)
        print('height = ', frame_height)
        print('Number of Frames = ', numframes)
        print('fps = ', vidfps)
        print('Length = ', vidlength)
        print('start = ', vidstart)

    return vidlength, numframes, vidfps, vidstart, frame_width, frame_height

def stanceSwingColors():
    stance_color = [0.95, 0.95, 0.95]
    swing_color = [0.15, 0.15, 0.15]
    return stance_color, swing_color

def addLegToPlot(f, a, ylev, footdown, footup, videoEnd=6.2):
    steps = []
    stepTimes = [0]
    stance_color, swing_color = stanceSwingColors()

    if footdown[0] < footup[0]:
        steps.append('u')
    else:
        steps.append('d')

    while len(footdown) > 0 and len(footup) > 0:
        if footdown[0] < footup[0]:
            steps.append('d')
            stepTimes.append(footdown[0])
            footdown = footdown[1:]
        else:
            steps.append('u')
            stepTimes.append(footup[0])
            footup = footup[1:]

    # deal with last step
    if len(footdown) > 0:
        steps.append('d')
        stepTimes.append(footdown[0])
    elif len(footup) > 0:
        steps.append('u')
        stepTimes.append(footup[0])

    lastStepTime = videoEnd
    stepTimes.append(lastStepTime)
    rectHeight = 1

    for i, step in enumerate(stepTimes[:-1]):
        if steps[i] == 'd':
            fc = stance_color
            ec = 'k'
        else:
            fc = swing_color
            ec = 'k'

        # ax.add_patch(Rectangle((1, 1), 2, 6))
        a.add_patch(Rectangle((stepTimes[i], ylev - rectHeight / 2), stepTimes[i + 1] - stepTimes[i], rectHeight,
                              edgecolor=None, facecolor=fc, fill=True, lw=1))

    return f, a


def parseFootLine(footline):
    return [int(x) / 1000 for x in footline.split(': ')[1].split()]


# get durations of stances and swings for a leg
# inputs = lists of times of foot-down (d) and foot-up (u)
# can get these lists from leg_dictionary, out of getDataFromFile
def getStancesSwings(d, u):
    stances = []
    swings = []

    # get stance durations
    if d[-1] > u[-1]:
        # last down is greater than last up
        # so ... going to ignore last down
        downs = d[:-1]
    else:
        downs = d

    for step in downs:
        upstep = u[np.argmax(u > step)]
        # print(step,upstep)
        duration = upstep - step
        stances.append(duration)

    # get swing durations
    if u[-1] > d[-1]:
        # last up is greater than last down
        # so ... going to ignore last up
        ups = u[:-1]
    else:
        ups = u

    for step in ups:
        downstep = d[np.argmax(d > step)]
        # print(step,downstep)
        duration = downstep - step
        swings.append(duration)

    stances = np.array(stances)
    swings = np.array(swings)

    return stances, swings


# given 2 legs ... get intervals between the down-steps of those legs
# input = lists of times of down-steps
def getStepIntervals(leg_down_1, leg_down_2):
    intervals = []

    if leg_down_1[-1] > leg_down_2[-1]:
        # last step of leg 1 is after leg 2 ... so we will ignore it
        leg_down_1 = leg_down_1[:-1]

    for step in leg_down_1:
        next_step = leg_down_2[np.argmax(leg_down_2 > step)]
        # print(step,upstep)
        duration = next_step - step
        intervals.append(duration)

    return intervals


# boxplot of stance or swing data
def plot_stance(leg_dict, leg_list, to_plot='stance', show_plot=True):
    plt.style.use('fivethirtyeight')
    box_data = []

    for leg in leg_list:
        stances_leg, swings_leg = getStancesSwings(np.array(leg_dict[leg]['d']), np.array(leg_dict[leg]['u']))
        if to_plot == 'swing':
            box_data.append(swings_leg)
            ylab = 'Swing Time (sec)'
        else:
            box_data.append(stances_leg)
            ylab = 'Stance Time (sec)'

    if show_plot:
        f, a = plot_box_data(leg_list, box_data, ylab, True)
    else:
        f, a = plot_box_data(leg_list, box_data, ylab, False)

    return box_data, f, a


def plot_box_data(leg_list, box_data, ylab, show_plot=True):
    f, a = plt.subplots(1, figsize=(len(leg_list), 4), facecolor='w')
    a.boxplot(box_data)
    a.set_xticklabels(leg_list)
    a.set_ylabel(ylab)
    a.set_xlabel('legs')
    a.set_facecolor('w')
    plt.subplots_adjust(bottom=0.13, left=0.13)
    if show_plot:
        plt.show()
    return f, a


def save_stance_swing(data_folder, leg_list, stance_data, swing_data):
    out_file = os.path.join(data_folder, 'stance_swing.csv')
    with open(out_file, 'w') as o:
        o.write('leg,datatype,data\n')
        for i, leg in enumerate(leg_list):
            o.write(leg + ',stance,' + ' '.join([str(round(x, 3)) for x in stance_data[i]]) + '\n')
            o.write(leg + ',swing,' + ' '.join([str(round(x, 3)) for x in swing_data[i]]) + '\n')
    return


def saveStepStats(leg_dict, leg_group='first'):
    stance_times = []
    swing_times = []
    gait_cycles = []
    duty_factors = []

    if leg_group == 'first':
        legs = ['L3', 'L2', 'L1', 'R1', 'R2', 'R3']
        leg_prefix = 'legs1-3_'
    else:
        legs = ['L4', 'R4']
        leg_prefix = 'leg4_'

    for leg in legs:
        stance_times.extend(leg_dict[leg]['stance_times'])
        swing_times.extend(leg_dict[leg]['swing_times'])
        gait_cycles.extend(leg_dict[leg]['gait_cycles'])
        duty_factors.extend(leg_dict[leg]['duty_factors'])

    ofile = leg_prefix + 'stance_times' + '.csv'
    o = open(ofile, 'w')
    for s in stance_times:
        o.write(str(s) + '\n')
    o.close()

    ofile = leg_prefix + 'swing_times' + '.csv'
    o = open(ofile, 'w')
    for s in swing_times:
        o.write(str(s) + '\n')
    o.close()

    ofile = leg_prefix + 'gait_cycles' + '.csv'
    o = open(ofile, 'w')
    for s in gait_cycles:
        o.write(str(s) + '\n')
    o.close()

    ofile = leg_prefix + 'duty_factors' + '.csv'
    o = open(ofile, 'w')
    for s in duty_factors:
        o.write(str(s) + '\n')
    o.close()


def find_nearest(num, arr):
    # given a number and an array of numbers
    # return the number from the array that is closest to the input number
    array = np.asarray(arr)
    idx = (np.abs(array - num)).argmin()
    return array[idx]


#### for swing combo
# functions to convert up and down lists for a leg into a vector of 1's (ups) and 0's (downs)

def valuesToProportions(dict_with_numerical_values):
    dict_with_proportional_values = {}
    
    # get total of all values in dictionary
    total_counts = 0
    
    for key in dict_with_numerical_values.keys():
        total_counts += dict_with_numerical_values[key]
    
    # calculate proportions of each count
    for key in dict_with_numerical_values.keys():
        dict_with_proportional_values[key] = dict_with_numerical_values[key] / total_counts
        
    return dict_with_proportional_values

def get_gait_combo_colors(leg_set = 'lateral'):

    if leg_set == 'rear':
        all_combos = ['stand','step','hop']
        plot_colors = get_plot_colors(len(all_combos))
        combo_colors = dict(zip(all_combos, plot_colors))
    else:
        all_combos = ['stand','pentapod','tetrapod_canonical','tetrapod_gallop', 'tetrapod_other',
                'tripod_canonical','tripod_other','other']
        plot_colors = get_plot_colors(len(all_combos))
        combo_colors = dict(zip(all_combos, plot_colors))
    return all_combos, combo_colors

def gait_style_plot(dict_list, clip_names, leg_set = 'lateral'):
    
    barWidth = 0.85
    num_bars = len(dict_list)
    fig_width = 1.5 * num_bars

    # set up colors
    if leg_set == 'rear':
        all_combos, combo_colors = get_gait_combo_colors('rear')
    else:
        all_combos, combo_colors = get_gait_combo_colors('lateral')

    f,ax = plt.subplots(1,1,figsize = (fig_width,5))

    for i, swing_combo_dict in enumerate(dict_list):

        combo_proportions = valuesToProportions(swing_combo_dict)

        for j, combo in enumerate(all_combos):

            if j == 0:
                bottom = 0

            if i == 0: # first dataset ... plot everything at 0 value to make labels for legend
                plt.bar(i, 0, bottom = bottom, color = combo_colors[combo],
                           edgecolor='white', width=barWidth, label=combo.replace('_',' '))

            if combo in swing_combo_dict.keys():

                plt.bar(i, combo_proportions[combo], bottom = bottom, color = combo_colors[combo],
                    edgecolor='white', width=barWidth)

                bottom += combo_proportions[combo]

    ax.set_xticks(np.arange(len(clip_names)))
    ax.set_xticklabels(clip_names, fontsize=12)

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc='upper left',
              bbox_to_anchor=(1,1), ncol=1)

    plt.ylabel('Proportion of frames', fontsize=16)
    
    return f, ax

# for ONE clip - plot steps with color-coded gait styles
# turn into a function where input is leg_set and movie_folder
def plotStepsAndGait(movie_folder, leg_set):

    # get step times for each leg for which we have data
    mov_data = os.path.join(movie_folder, 'mov_data.txt')
    up_down_times, latest_event = getUpDownTimes(mov_data)

    # quality control on up_down_times
    qcUpDownTimes(up_down_times)

    if leg_set == 'rear':
        legs = get_leg_combos()['legs_4']
    else:
        legs = get_leg_combos()['legs_lateral']

    # Get all frame times for this movie
    frame_times = get_frame_times(movie_folder)

    # trim frame_times to only include frames up to last recorded event
    last_event_frame = np.min(np.where(frame_times > latest_event*1000))
    frame_times_with_events = frame_times[:last_event_frame]

    # get leg matrix
    leg_matrix = make_leg_matrix(legs, up_down_times, frame_times_with_events)
    legs = np.array(legs)

    # set up colors
    if leg_set == 'rear':
        all_combos, combo_colors = get_gait_combo_colors('rear')
    else:
        all_combos, combo_colors = get_gait_combo_colors('lateral')

    # make a vector of colors for each frame, depending on combination of swinging legs
    swing_color_vector = []

    for col_ind in np.arange(np.shape(leg_matrix)[1]):
        one_indices = np.where(leg_matrix[:, col_ind] == 1)
        swinging_legs = legs[one_indices]
        swinging_leg_combo = '_'.join(sorted(swinging_legs))
        gait_style = get_swing_categories(swinging_leg_combo, leg_set)
        combo_color = combo_colors[gait_style]
        swing_color_vector.append(combo_color)

    # set up plot
    # get colors for stance and swing
    stance_color, swing_color = stanceSwingColors()

    # definitions for the axes
    # left, bottom, width, height
    if leg_set == 'rear':
        rect_steps = [0.07, 0.07, 1, 0.5]
        rect_gaits = [0.07, 0.6, 1, 0.2]
        fig_height = int(len(legs))
    else:
        rect_steps = [0.07, 0.07,  1, 0.6]
        rect_gaits = [0.07, 0.70,  1, 0.1]
        fig_height = int(len(legs) * 0.7)

    stepplot_colors = {1: swing_color, 0: stance_color}
    gait_x = frame_times_with_events/1000
    
    fig = plt.figure(figsize=(10,fig_height))
    steps = plt.axes(rect_steps)

    bar_width = gait_x[1] - gait_x[0]      

    for i, leg in enumerate(legs):
        for j, x in enumerate(gait_x):
            steps.barh(i+1, bar_width, height=1, left=j*bar_width, 
                    color = stepplot_colors[leg_matrix[i, j]])

    steps.set_ylim([0.5, len(legs)+0.5])
    steps.set_xlabel('Time (sec)', fontsize=16)
    steps.set_yticks(np.arange(len(legs))+1)
    steps.set_yticklabels(legs, fontsize=16)
    steps.set_ylabel('legs', fontsize=16)
    steps.set_frame_on(False)

    gaits = plt.axes(rect_gaits)

    for i,x in enumerate(gait_x):
        gaits.barh(1, bar_width, height=0.8, left=i*bar_width, color = swing_color_vector[i])

    gaits.set_ylabel('gait', fontsize=16)

    gaits.set_xlim([0, gait_x[-1]])
    steps.set_xlim([0, gait_x[-1]])

    gaits.set_yticks([])
    gaits.set_xticks([])
    gaits.set_frame_on(False)
    
    fig.suptitle(movie_folder.split(os.sep)[-1], fontsize=24)
    
    return fig

def define_swing_categories():
    leg_combo_keys = ['tripod_canonical', 
                'tetrapod_canonical', 
                'tetrapod_gallop',
                ]    
    # combinations should be sorted by leg name!
    leg_combo_values = [['L1_L3_R2', 'L2_R1_R3'],
                        ['L1_R2', 'L2_R3', 'L3_R1', 'L2_R1', 'L3_R2', 'L1_R3'],
                        ['L1_R1', 'L2_R2', 'L3_R3']]    
    swing_categories = dict(zip(leg_combo_keys, leg_combo_values))
    return swing_categories

def rearCombos(rearleg_swing_counts):
    rear_combos = {}
    
    if 'none' in rearleg_swing_counts.keys():  
        rear_combos['stand'] = rearleg_swing_counts['none']
    if 'L4_R4' in rearleg_swing_counts.keys():
        rear_combos['hop'] = rearleg_swing_counts['L4_R4']
    
    rear_combos['step'] = 0
    if 'L4' in rearleg_swing_counts.keys():
        rear_combos['step'] += rearleg_swing_counts['L4']
    if '$4' in rearleg_swing_counts.keys():
        rear_combos['step'] += rearleg_swing_counts['R4']
            
    return rear_combos

def get_leg_swing_combos(movie_folder, leg_set = 'lateral'): # lateral or rear

    leg_combos = get_leg_combos()
    if leg_set == 'rear':
        legs = leg_combos['legs_4']
    else:
        legs = leg_combos['legs_lateral']
    
    # set up variables to collect the data we need
    leg_swing_combos = {}
    total_frames = 0

    # get frame_times for this movie (in milliseconds, e.g. [0 33 66 100 .... ])
    frame_times = get_frame_times(movie_folder)
    total_frames += len(frame_times)

    # get dictionary of up & down timing for this video clip
    # keys = leg['u'] or leg['d'] where leg is in ['L4','L3','L2','L1' (or rights)]
    up_down_times, latest_event = getUpDownTimes(os.path.join(movie_folder, 'mov_data.txt'))

    # get matrix of up (1's) and down (0's) data for all legs
    # rows = legs
    # columns = frames of video
    leg_matrix = make_leg_matrix(legs, up_down_times, frame_times)

    # get dictionary of #frames swinging for different combinations of legs 
    leg_swing_counts = get_leg_swing_counts(leg_matrix, leg_set)

    # get counts of #frames in each type of swing category
    for combo in leg_swing_counts.keys():
        swing_category = get_swing_categories(combo, leg_set)
        if swing_category in leg_swing_combos.keys():
            leg_swing_combos[swing_category] += leg_swing_counts[combo]
        else:       
            leg_swing_combos[swing_category] = leg_swing_counts[combo]
        
    return leg_swing_combos

def combineDictionariesWithCommonKeys(dict_list):
    combined_dict = {}

    for d in dict_list:
        for k in d.keys():
            if k in combined_dict.keys():
                combined_dict[k] += d[k]
            else:
                combined_dict[k] = d[k]
    
    return combined_dict

def get_swing_categories(swing_combination, leg_set = 'lateral'):
    
    if leg_set == 'rear':

        if swing_combination == 'none' or swing_combination == '':
            gait_style = 'stand'
        elif swing_combination == 'L4_R4':
            gait_style = 'hop'
        else: # L4 or R4
            gait_style = 'step'

    else:

        # swing_combination is a sorted string of leg names, separated by underscore
        # e.g. 'L1_L3_R2' or 'L1_R3' (all L's comes before R's)    
        swing_categories = define_swing_categories() # these are tripod and tetrapod groups
        
        # how many legs are swinging?
        if swing_combination == 'none' or swing_combination == '':
            num_legs_swinging = 0
            swinging_legs = ''
        else:
            swinging_legs = swing_combination.split('_')    
            num_legs_swinging = len(swinging_legs)
        
        if num_legs_swinging == 0:
            gait_style = 'stand'
        elif num_legs_swinging == 1:
            gait_style = 'pentapod'
        elif num_legs_swinging == 2: # tetrapod!
            if swing_combination in swing_categories['tetrapod_canonical']:
                gait_style = 'tetrapod_canonical'
            elif swing_combination in swing_categories['tetrapod_gallop']:
                gait_style = 'tetrapod_gallop'
            else:
                gait_style = 'tetrapod_other'
        elif num_legs_swinging == 3: # tripod!
            if swing_combination in swing_categories['tripod_canonical']:
                gait_style = 'tripod_canonical'
            else:
                gait_style = 'tripod_other'
        else:
            gait_style = 'other' # 4 or more 
    
    return gait_style


def get_frame_times(movie_folder):
    video_file = getMovieFromFileList(movie_folder)
    vid = cv2.VideoCapture(os.path.join(movie_folder, video_file))
    vidlength, numframes, vidfps, vidstart, frame_width, frame_height = getVideoStats(vid, False)
    vid.release()
    frame_times = np.array([int(x) for x in np.linspace(0, vidlength * 1000, int(numframes))])
    return frame_times


def uds_to_ones(ups, downs, leg_vector, frame_times):
    # ups and downs are lists of equal length
    # each up should be < each down
    # the interval between up and down is a swing ... fill in this interval with 1's

    if len(ups) == len(downs):
        for i, up in enumerate(ups):
            down = downs[i]
            leg_vector[np.where(np.logical_and(frame_times > up, frame_times < down))] = 1
    else:
        print('Problem - # ups != # downs !')

    return leg_vector


def up_down_times_to_binary(downs, ups, frame_times):
    # convert list of leg-down and leg-up to vector of 0's (for stance), 1's (for swing)
    # first, need to deal with differences in up/down order ... convert them to just swings ududud
    # and add swing data (i.e. 1's) at beginning and/or end of leg_vector if necessary

    # make an empty vector for this leg
    leg_vector = np.zeros(len(frame_times))

    # leg patterns can be dududu, ududu, dududud, ududud
    # swing data at beginning and end depends on whether d/u first or last
    # figure this out, and then convert to ududud (just swings)

    if downs[0] < ups[0] and ups[-1] > downs[-1]:  # input is dududu
        # ... fill in start to first down with 1's
        # ... fill in last up to end with 1's
        # print('dududu ... filling in start frames with ones, end frames with ones, and omitting first down and last up')

        leg_vector[np.where(frame_times < downs[0])] = 1
        leg_vector[np.where(frame_times > ups[-1])] = 1

        downs = downs[1:]
        ups = ups[:-1]

    elif downs[0] > ups[0] and ups[-1] > downs[-1]:  # input is ududu
        # ... fill in last up to end with 1's
        # print('ududu ... filling in end frames with ones, and omitting last up')

        leg_vector[np.where(frame_times > ups[-1])] = 1

        ups = ups[:-1]

    elif downs[0] < ups[0] and downs[-1] > ups[-1]:  # input is dududud
        # ... fill in start to first down with 1's
        # print('dududud ... filling in start frames with ones, and omitting first down')

        leg_vector[np.where(frame_times < downs[0])] = 1

        downs = downs[1:]

    elif downs[0] > ups[0] and downs[-1] > ups[-1]:  # input is ududud
        # ... start and end with down, no need to fill in swing data
        # print('ududud ... go ahead and find swings')
        pass
    else:
        print('uh oh no pattern match')
        sys.exit()

    # convert each swing interval to 1's
    leg_vector = uds_to_ones(ups, downs, leg_vector, frame_times)

    return leg_vector

def make_leg_matrix(legs, up_down_times, frame_times):
    # Build a matrix:
    # rows = vector of swings (1's) and stances (0's) for each leg
    # columns = each frame of video clip

    # make empty matrix
    leg_matrix = np.zeros([len(legs), len(frame_times)])

    # fill up each row with leg data
    for i, leg in enumerate(legs):
        # print(leg)
        ups = np.array(up_down_times[leg]['u'])
        downs = np.array(up_down_times[leg]['d'])
        leg_vector = up_down_times_to_binary(downs, ups, frame_times / 1000)
        leg_matrix[i, :] = leg_vector

    return leg_matrix

def get_leg_swing_counts(leg_matrix, leg_set = 'lateral'):
    # function to get dictionary of #frames swinging for combinations of legs
    # keys = leg_combo (e.g. 'L1_R2')
    # values = number of frames where that combination of legs = swinging simultaneously

    leg_combos = get_leg_combos()
    if leg_set == 'rear':
        legs = leg_combos['legs_4']
    else:
        legs = leg_combos['legs_lateral']

    legs = np.array(legs)

    # get number of frames
    num_cols = np.shape(leg_matrix)[1]

    # set up leg_swing_counts dictionary
    # set count of where no legs are swinging to zero
    leg_swing_counts = {}
    leg_swing_counts['none'] = 0

    for col_ind in np.arange(num_cols):

        # in this column (frame), find indices of legs that are swinging (i.e. equal to 1)
        one_indices = np.where(leg_matrix[:, col_ind] == 1)

        # convert indices to leg names
        swinging_legs = legs[one_indices]

        # combine legs into a single string to use as a key
        swinging_leg_key = '_'.join(sorted(swinging_legs))

        # if this frame has no legs swinging, add one to the 'none' count
        if len(swinging_leg_key) == 0:
            leg_swing_counts['none'] += 1

        # if this combo is not yet in the dictionary, 
        else:
            # if this combo is already in the dictionary, add one to its count
            if swinging_leg_key in leg_swing_counts.keys():
                leg_swing_counts[swinging_leg_key] += 1
            # if this combo is not yet in the dictionary, set its count to 1
            else:
                leg_swing_counts[swinging_leg_key] = 1

    return leg_swing_counts


def add_counts_to_dictionary(new_data, existing_dictionary):
    # add counts from new dictionary to old dictionary
    # new_data is a dictionary of keys => counts
    # existing has keys => counts

    # for each key of new_data 
    for k in new_data.keys():

        # if this key is in existing_dictionary, add to existing counts
        if k in existing_dictionary.keys():
            existing_dictionary[k] += new_data[k]

        # if this key is not in existing_dictionary, make an entry
        else:
            existing_dictionary[k] = new_data[k]

    # return updated dictionary
    return existing_dictionary

# convert step data for a single clip into a dataframe
def stepDataToDf(foldername, fname):
    fpath = os.path.join(foldername, fname)
    fileTest(fpath)  # to test if file exists before trying to open it
    df = pd.read_csv(fpath, index_col=None)

    # add column that contains folder name
    num_rows = df.shape[0]
    exp_column = [foldername] * num_rows
    df['clip'] = exp_column

    return df


# given multiple folders, combine step data from each folder into a dataframe
def foldersToDf(folder_list, fname):
    if len(folder_list) == 1:
        step_data = stepDataToDf(folder_list[0], fname)
    else:
        step_data = stepDataToDf(folder_list[0], fname)
        folder_list = folder_list[1:]
        for folder in folder_list:
            df = stepDataToDf(folder, fname)
            step_data = pd.concat([step_data, df])

    return step_data


# given a folder that contains multiple folders, each with step data
# combine step data from each folder into a dataframe
def experimentToDf(experiment_directory, fname):
    os.chdir(experiment_directory)
    # list directories in this folder
    clip_directories = listDirectories()

    clip_list = sorted(selectMultipleFromList(clip_directories))
    df = foldersToDf(clip_list, fname)
    os.chdir('../')
    return df


# given a dataframe containing step data
# return metachronal lag (time between swings of hindlimbs and forelimbs)
#     swing of foreleg step seen AFTER midleg swing AFTER hindleg swing!
# and return normalized metachronal lag (normalized to hindlimb period)
def get_metachronal_lag(df):
    metachronal_lag = []  # initialize empty list
    normalized_metachronal_lag = []  # initialize empty list
    clips = np.unique(df['clip'].values)

    hind_legs = ['L3', 'R3']
    mid_legs = ['L2', 'R2']
    fore_legs = ['L1', 'R1']

    # go through each clip
    for clip in clips:
        clip_slice = df[df['clip'] == clip]

        # go through each hind_leg
        for leg_index, hind_leg in enumerate(hind_legs):
            mid_leg = mid_legs[leg_index]
            fore_leg = fore_legs[leg_index]

            hind_steps = clip_slice[clip_slice.ref_leg == hind_leg]['down_time'].values
            hind_swings = clip_slice[clip_slice.ref_leg == hind_leg]['up_time'].values
            hind_periods = clip_slice[clip_slice.ref_leg == hind_leg]['gait_cycle'].values
            mid_steps = clip_slice[clip_slice.ref_leg == mid_leg]['down_time'].values
            fore_steps = clip_slice[clip_slice.ref_leg == fore_leg]['down_time'].values
            fore_swings = clip_slice[clip_slice.ref_leg == fore_leg]['up_time'].values

            # go through hind_steps
            for i, step in enumerate(hind_steps):

                # get gait_cycle associated with this step
                hind_leg_period = hind_periods[i]

                # get swing_time associated with this step
                hind_swings_after_step = hind_swings[np.where(hind_swings > step)]
                if len(hind_swings_after_step) == 0:
                    break
                else:
                    hind_swing_time = hind_swings_after_step[0]

                # get mid_leg step AFTER the hind_leg step
                mid_steps_after_hind_step = mid_steps[np.where(mid_steps > step)]
                if len(mid_steps_after_hind_step) == 0:
                    break
                else:
                    mid_step_time = mid_steps_after_hind_step[0]

                # get fore_leg step AFTER this mid leg step
                fore_steps_after_mid_step = fore_steps[np.where(fore_steps > mid_step_time)]
                if len(fore_steps_after_mid_step) == 0:
                    break
                else:
                    fore_step_time = fore_steps_after_mid_step[0]

                # get fore_leg SWING after this fore_leg step
                fore_swings_after_fore_steps = fore_swings[np.where(fore_swings > fore_step_time)]
                if len(fore_swings_after_fore_steps) == 0:
                    break
                else:
                    fore_swing_time = fore_swings_after_fore_steps[0]

                # passed all the tests ... find the metachronal lag for this step
                lag = fore_swing_time - hind_swing_time # could also do fore_step_time - 
                if lag > 0:  # sometimes hindstep duration is super long, and foreleg swings before hindleg does
                    metachronal_lag.append(np.round(lag, 3))
                    normalized_metachronal_lag.append(np.round(lag / hind_leg_period, 3))

    return metachronal_lag, normalized_metachronal_lag

# prompt to remove the folder containing individual frames saved from a clip
def removeFramesFolder(data_folder):
    frames_folder = os.path.join(data_folder, data_folder + '_frames')
    fileList = glob.glob(os.path.join(data_folder, '*'))
    if frames_folder in fileList:
        selection = input('Remove frames folder? (y) or (n): ')
        if selection == 'y':
            print(' ... removing ' + frames_folder + '\n')
            shutil.rmtree(frames_folder)

# check to see if we have the beginning and ending frames to calculate speed
# if we do not have them, make them!
def saveSpeedFrames(data_folder, movie_info):
    
    # check to see if we have the beginning and ending frames to calculate speed
    haveSpeedFrames = False
    beginning_speed_frame = os.path.join(data_folder,'beginning_speed_frame.png')
    ending_speed_frame = os.path.join(data_folder,'ending_speed_frame.png')
    fileList = glob.glob(os.path.join(data_folder, '*'))
    
    if beginning_speed_frame in fileList and ending_speed_frame in fileList:
        print(' ... found the speed frames in ' + data_folder)
    
    else:
        print(' ... no speed frames yet - saving them now!')
        print('speed frame range ' + movie_info['speed_framerange'])

        if movie_info['speed_start'] > 0 and movie_info['speed_end'] > 0:
            
            beginning_speed_range = int(movie_info['speed_start'] * 1000)
            ending_speed_range = int(movie_info['speed_end'] * 1000)

        # elif movie_info['speed_framerange'] == 'none':
        else:
            print(' ... no speed boundaries available, just getting first and last frames ...')
            beginning_speed_range = int(movie_info['start_frame'] * 1000)
            ending_speed_range = int(movie_info['end_frame'] * 1000)

        need_beginning = True
        need_ending = True

        vid = cv2.VideoCapture(os.path.join(data_folder, movie_info['movie_name']))
        while (vid.isOpened()):
        
            ret, frame = vid.read()
            if ret: # found a frame
                frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC))
                if frameTime >= beginning_speed_range and need_beginning:
                    print('saving ' + beginning_speed_frame + ' at ' + str(movie_info['speed_start']))
                    cv2.imwrite(beginning_speed_frame, frame)
                    need_beginning = False
                if frameTime >= ending_speed_range and need_ending:
                    print('saving ' + ending_speed_frame + ' at ' + str(movie_info['speed_end']))
                    cv2.imwrite(ending_speed_frame, frame)
                    need_ending = False
            else: # no frame here
                break

        vid.release()
    
    return

def dataAfterColon(line):
    # given a string with ONE colon
    # return data after the colon
    return line.split(':')[1].replace(' ','')

def getRangeFromText(text_range):
    beginning = float(text_range.split('-')[0])
    ending = float(text_range.split('-')[1])
    return beginning, ending

# get information about a clip (e.g. timing, tardigrade size, tardigrade speed) from mov_data.txt
def getMovieInfo(data_folder):

    mov_datafile = os.path.join(data_folder, 'mov_data.txt')

    movie_info = {}
    movie_info['movie_name'] = ''
    movie_info['movie_length'] = 0
    movie_info['analyzed_framerange'] = ''
    movie_info['start_frame'] = 0
    movie_info['end_frame'] = 0
    movie_info['speed_framerange'] = ''
    movie_info['speed_start'] = 0
    movie_info['speed_end'] = 0
    movie_info['tardigrade_width'] = 0
    movie_info['tardigrade_length'] = 0
    movie_info['field_width'] = 0
    movie_info['distance_traveled'] = 0
    movie_info['tardigrade_speed'] = 0

    with open(mov_datafile, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('MovieName'):
                movie_info['movie_name'] = dataAfterColon(line)
            if line.startswith('Length'):
                movie_info['movie_length'] = float(dataAfterColon(line))
            if line.startswith('Analyzed Frames'):
                frameRange = dataAfterColon(line)
                movie_info['analyzed_framerange'] = frameRange
                movie_info['start_frame'], movie_info['end_frame'] = getRangeFromText(frameRange)
            if line.startswith('Speed'):
                if 'none' in line:
                    movie_info['speed_framerange'] = 'none'
                    movie_info['speed_start'], movie_info['speed_end'] = (0,0)
                else:
                    speedRange = dataAfterColon(line)
                    movie_info['speed_framerange'] = speedRange
                    movie_info['speed_start'], movie_info['speed_end'] = getRangeFromText(speedRange)
            if line.startswith('Field'):
                movie_info['field_width'] = float(dataAfterColon(line))
            if line.startswith('Tardigrade Width'):
                movie_info['tardigrade_width'] = float(dataAfterColon(line))
            if line.startswith('Tardigrade Length'):
                movie_info['tardigrade_length'] = float(dataAfterColon(line))
            if line.startswith('Distance Traveled'):
                movie_info['distance_traveled'] = float(dataAfterColon(line))
            if line.startswith('Tardigrade Speed'):
                if 'none' not in line:
                    movie_info['tardigrade_speed'] = float(dataAfterColon(line))
                else:
                    movie_info['tardigrade_speed'] = dataAfterColon(line)

        # if no information for 'Analyzed Frames', retrieve it from the movie
        if movie_info['start_frame'] == 0 or movie_info['end_frame'] == 0:
            print('No info about analyzed frame times ... getting that now ... ')
            first_frame, last_frame = getStartEndTimesFromMovie(data_folder, movie_info)
            print('   ... start is ' + str(first_frame) + ', end is ' + str(last_frame))
            movie_info['start_frame'], movie_info['end_frame'] = first_frame, last_frame
            movie_info['analyzed_framerange'] = str(first_frame) + '-' + str(last_frame)

    return movie_info

# from a treatment folder containing folders of clips
# return a dictionary containing size and speed information for each clip
def sizeAndSpeed(treatment_dir, clip_folders, scale = 1.06): # use 0 if do not want to convert from pix to µm
    # scale is calculated by measuring the # pixels in the field of view of the scope
    # and measuring the # of pixels / µm via a micrometer
    # and using these to figure out what the distance of the field of view is
    # in Jean's lab at 10X, field of view is 1.06 mm
    # on Olympus CH30 at 10X, field of view is 1.1 mm

    if scale > 0:
        print('... converting from pixels to micrometers')

    # set up a dictionary to contain size and speed data for each clip
    size_speed = {}
        
    # go through all the selected clips and add data to size_speed dicionary
    for clip in clip_folders:
        data_folder = os.path.join(treatment_dir, clip)
        
        # get dictionary of  info about movie from mov_data.txt
        movie_info = getMovieInfo(data_folder)

        if scale > 0:
            pix_to_um_conversion = (scale * 1000) / movie_info['field_width']
            print('... for ' + clip + ', conversion is ' + str(np.around(pix_to_um_conversion,2)) + ' micrometers per pixel.')
        else:
            pix_to_um_conversion = 1   

        # what is the time interval used to calculate speed?
        size_speed[clip] = {}
        size_speed[clip]['analyzed_time'] = movie_info['end_frame'] - movie_info['start_frame']

        # get the field width, so we can convert pixels to µm
        size_speed[clip]['field_width'] = movie_info['field_width']

        # get distance and length, and convert to pixels
        size_speed[clip]['tardigrade_length'] = movie_info['tardigrade_length'] * pix_to_um_conversion
        size_speed[clip]['distance_traveled'] = movie_info['distance_traveled'] * pix_to_um_conversion

        # area approximate as an ellipse. (length/2 * width/2 * pi)
        tardigrade_area = (movie_info['tardigrade_width'] * pix_to_um_conversion) / 2 * (pix_to_um_conversion* movie_info['tardigrade_length']) / 2 * np.pi
        size_speed[clip]['tardigrade_area'] = tardigrade_area
        
        # speed is distance / time ... so we can use pix_to_um_conversion to scale speed too
        size_speed[clip]['tardigrade_speed'] = movie_info['tardigrade_speed'] * pix_to_um_conversion

    return size_speed

def getStartEndTimesFromMovie(data_folder, movie_info):
    frameTimes = []

    vid = cv2.VideoCapture(os.path.join(data_folder, movie_info['movie_name']))
    while (vid.isOpened()):
    
        ret, frame = vid.read()
        if ret: # found a frame
            frameTime = int(vid.get(cv2.CAP_PROP_POS_MSEC))
            frameTimes.append(frameTime)

        else:
            break
    vid.release()

    frameTimes = [float(x)/1000 for x in frameTimes if x > 0]
    return frameTimes[0], frameTimes[-1]

def updateMovieData(data_folder, movie_info):

    # make a backup copy of mov_data.txt
    movie_datafile = os.path.join(data_folder, 'mov_data.txt')
    print('\nUpdating ' + movie_datafile)

    backup_datafile = os.path.join(data_folder, 'backup_mov_data.txt' )
    os.rename(movie_datafile, backup_datafile)

    print_now = False

    o = open(movie_datafile, 'w')
    o.write('MovieName: ' + movie_info['movie_name'] + '\n' )
    o.write('Length: ' + str(movie_info['movie_length']) + '\n' )
    o.write('Analyzed Frames: ' + movie_info['analyzed_framerange'] + '\n' )
    o.write('Tardigrade Width: ' + str(movie_info['tardigrade_width']) + '\n')
    o.write('Tardigrade Length: ' + str(movie_info['tardigrade_length']) + '\n')
    o.write('Field of View: ' + str(movie_info['field_width']) + '\n')
    
    if movie_info['speed_start'] > 0 and movie_info['speed_end'] > movie_info['speed_start']:
        o.write('Speed Frames: ' + movie_info['speed_framerange'] + '\n' )
        time_elapsed = movie_info['speed_end'] - movie_info['speed_start']
        movie_info['tardigrade_speed'] = movie_info['distance_traveled'] / time_elapsed
    else:
        o.write('Speed Frames: none' + '\n' )

    o.write('Distance Traveled: ' + str(movie_info['distance_traveled']) + '\n')
    o.write('Tardigrade Speed: ' + str(movie_info['tardigrade_speed']) + '\n')
    o.write('\n')

    with open(backup_datafile, 'r') as f:
        for line in f:
            if 'Data for' in line:
                print_now = True
            if print_now == True:
                o.write(line)

    o.close()

# in a pair of experiments loaded into df1 and df2 
# plot step parameters for each experiment for a set of legs to compare

# error bars for stance and swing duration:
# choose from std or sem (could also use 95%CI but need to code that)
# sem is really small, b/c sample size (every single step for all legs being compared) is so big
def compare_step_parameters(groups, dataframes, legs):
    
    df1, df2 = dataframes

    stance_color, swing_color = stanceSwingColors()

    # get stances and swings from the two dataframes
    stances_1 = df1[df1['ref_leg'].isin(legs)]['stance_time'].values
    stances_2 = df2[df2['ref_leg'].isin(legs)]['stance_time'].values
    stances = [np.mean(stances_1),np.mean(stances_2)]
    # stance_err = [np.std(stances_1),np.std(stances_2)]
    stance_err = [sem(stances_1),sem(stances_2)]

    swings_1 = df1[df1['ref_leg'].isin(legs)]['swing_time'].values
    swings_2 = df2[df2['ref_leg'].isin(legs)]['swing_time'].values
    swings = [np.mean(swings_1),np.mean(swings_2)]
    # swing_err = [np.std(swings_1),np.std(swings_2)]
    swing_err = [sem(swings_1),sem(swings_2)]

    # get gait cycles from the two dataframes
    gait_cycles_1 = df1[df1['ref_leg'].isin(legs)]['gait_cycle'].values
    gait_cycles_2 = df2[df2['ref_leg'].isin(legs)]['gait_cycle'].values
    gait_cycles = [gait_cycles_1,gait_cycles_2]

    # get duty factors from the two dataframes
    duty_factors_1 = df1[df1['ref_leg'].isin(legs)]['duty_factor'].values
    duty_factors_2 = df2[df2['ref_leg'].isin(legs)]['duty_factor'].values
    duty_factors = [duty_factors_1,duty_factors_2]

    # get metachronal lags from the two dataframes
    ml1, nml1 = get_metachronal_lag(df1)
    ml2, nml2 = get_metachronal_lag(df2)
    ml = [ml1,ml2]
    nml = [nml1,nml2]

    # set up a figure
    fig = plt.figure(figsize=(8,6), dpi=300, constrained_layout = True)

    # Stance and Swing duration
    ax1 = fig.add_subplot(321)
    ax1.barh(groups, stances, align='center', height=.25, xerr = stance_err,
             color=stance_color, label='stances')
    ax1.barh(groups, swings, align='center', height=.25, left=stances, xerr = swing_err,
             color=swing_color, label='swings')
    ax1.set_xlabel('Stance and Swing time (sec)')
    ax1.set_yticks(np.array(range(len(groups))))
    ax1.set_yticklabels(groups, fontsize=10)
    ax1.set_facecolor('lightsteelblue') #lightslategrey skyblue darkseagreen lightsteelblue

    # Gait_cycle (aka leg period)
    ax3 = fig.add_subplot(323)
    bp3 = ax3.boxplot(gait_cycles, vert = False)
    bp3 = bw_boxplot(bp3)
    ax3.set_yticks(np.array(range(len(groups)))+1)
    ax3.set_yticklabels(groups, fontsize=10)
    ax3.set_xlabel('Leg Period (sec)')

    # Duty Factor
    ax5 = fig.add_subplot(325)
    bp5 = ax5.boxplot(duty_factors, vert = False)
    bp5 = bw_boxplot(bp5)
    ax5.set_yticks(np.array(range(len(groups)))+1)
    ax5.set_yticklabels(groups, fontsize=10)
    ax5.set_xlabel('Duty factor')

    # Metachronal lag
    ax2 = fig.add_subplot(322)
    bp2 = ax2.boxplot(ml, vert = False)
    bp2 = bw_boxplot(bp2)
    ax2.set_yticks(np.array(range(len(groups)))+1)
    ax2.set_yticklabels(groups, fontsize=10)
    ax2.set_xlabel('Metachronal lag (sec)')

    # Normalized metachronal lag
    ax4 = fig.add_subplot(324)
    bp4 = ax4.boxplot(nml, vert = False)
    bp4 = bw_boxplot(bp4)
    ax4.set_yticks(np.array(range(len(groups)))+1)
    ax4.set_yticklabels(groups, fontsize=10)
    ax4.set_xlabel('Normalized metachronal lag')

    plt.show()

# format colors of a boxplot object
def formatBoxPlots(bp, boxColors=[], medianColors=[], flierColors=[]):
    
    # if no colors specified
    if len(boxColors) == 0:
        boxColors = ['tab:blue'] * len(bp['boxes'])
        medianColors = ['white'] * len(bp['boxes'])
        flierColors = ['lightsteelblue'] * len(bp['boxes'])
    elif len(boxColors) == 1:
        boxColors = boxColors * len(bp['boxes'])
        medianColors = medianColors * len(bp['boxes'])
        flierColors = flierColors * len(bp['boxes'])
        
    baseWidth = 2
    for n,box in enumerate(bp['boxes']):
        box.set( color=boxColors[n], linewidth=baseWidth)

    for n,med in enumerate(bp['medians']):
        med.set( color=medianColors[n], linewidth=baseWidth)

    bdupes=[]
    for i in boxColors:
        bdupes.extend([i,i])

    boxColors = bdupes
    for n,whisk in enumerate(bp['whiskers']):
        #whisk.set( color=(0.1,0.1,0.1), linewidth=2, alpha = 0.5)
        whisk.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)

    for n,cap in enumerate(bp['caps']):
        cap.set( color=boxColors[n], linewidth=baseWidth, alpha = 0.5)
        
    # fliers
    for n, flier in enumerate(bp['fliers']): 
        flier.set(marker ='.', color = flierColors[n], alpha = 0.5) 

    return bp

# black and white boxplot
def bw_boxplot(bp):
    for box in bp['boxes']:
        box.set(color ='k', linewidth = 2, linestyle ="-")

    for whisker in bp['whiskers']:
        whisker.set(color ='k', linewidth = 2, linestyle ="-")

    for cap in bp['caps']:
        cap.set(color ='k', linewidth = 2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='k', linewidth = 2)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='.', color ='#e7298a', alpha = 0.5)
        
    return bp

# get plot colors
def get_plot_colors(num_colors=9, palette = 'default'):
    # see https://matplotlib.org/stable/gallery/color/named_colors.html
    if palette == 'tab':
        plot_colors = np.array(['tab:orange','tab:green','tab:purple','tab:red',
                       'tab:blue', 'tab:cyan', 'tab:pink', 'tab:olive', 'black'])
    else:
        plot_colors = np.array(['firebrick','gold','forestgreen','steelblue',
                   'darkviolet','darkorange', 'lawngreen', 'gainsboro', 'black'])

    if num_colors > len(plot_colors):
        print('too many colors')
        return plot_colors
    else:
        return plot_colors[:num_colors]