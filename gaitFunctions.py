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
import re


def getTrackingConfidence(problem_frames, difference_threshold, printme = False):
    tracking_confidence = round( 100 - (sum(problem_frames) / len(problem_frames)) * 100 , 2)
    if printme:
        print('Tracking confidence at a pixel threshold of ' + str(difference_threshold) + ' is ' + str(tracking_confidence) + '%')
        if tracking_confidence < 90:
            print('... you might want to try running trackCritter again with a different pixel threshold')
            if difference_threshold < 15:
                new_threshold = 25
            else:
                new_threshold = 12
            print('... try a threshold of ' + str(new_threshold) + ' instead of ' + str(difference_threshold))
    return tracking_confidence

def individualFromClipname(clipname):
    res = re.findall(r"[abcdefghijklmnopqrstuvwxyz_]+", clipname)[0]
    individual = clipname.split(res)[0]
    return individual

def one_runs(a):
    '''
    
    From a 1-d array (vector) of zeros and ones
    Find the indices where each 'bout' of ones starts and stops

    Parameters
    ----------
    a : numpy array
        a vector of zeros and ones.

    Returns
    -------
    ranges : numpy array
        start and end indices of each 'run' of ones within a.

    '''
    # Create an array that is 1 where a is 1, and pad each end with an extra 0.
    isone = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isone))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

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

def save_stance_figures(movie_file, up_down_times):
    legs = get_leg_combos()['legs_all']
    for x in ['stance', 'swing']:
        box_data, f, a = plot_stance(up_down_times, legs, x, False)
        fname = movie_file.split('.')[0] + x + '_plot.png'
        plt.savefig(fname)

# def save_leg_figures(movie_file, up_down_times, video_end):
#     leg_combos = get_leg_combos()
#     for legs in leg_combos.keys():
#         f, a = plot_legs(up_down_times, leg_combos[legs], video_end, False)
#         figname = movie_file.split('.')[0] + '_' + '_'.join(leg_combos[legs]) + '.png'
#         plt.savefig(figname)

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

# plot_legs is deprecated
# def plot_legs(legDict, legs, video_end, show=True):
#     leg_yvals = list(range(len(legs)))

#     # start a plot for the data
#     figheight = len(legs)
#     (f, a) = plt.subplots(1, figsize=(10, figheight))  # set height on # of legs

#     # add a leg to the plot
#     for i, leg in enumerate(legs):
#         yval = leg_yvals[i]
#         fd = legDict[leg]['d']
#         fu = legDict[leg]['u']
#         f, a = addLegToPlot(f, a, yval, fd, fu, video_end)

#     # show the plot
#     a.set_xlim([0, video_end])
#     y_buff = 0.5
#     a.set_ylim([leg_yvals[0] - y_buff, leg_yvals[-1] + y_buff])
#     a.set_yticks(leg_yvals)
#     a.set_yticklabels(legs)
#     a.set_xlabel('Time (sec)')
#     a.set_ylabel('Legs')
#     plt.subplots_adjust(bottom=0.3)
#     if show:
#         plt.show()
    # return f, a


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


def getUpDownTimes(mov_data):
    '''
    a function to take a mov_data dictionary and get up/down timing for a given leg

    Parameters
    ----------
    mov_data : dictionary
        keys = leg_state (e.g. 'L1_up', or 'R4_down')
        values = a string of integers representing timing in milliseconds

    Returns
    -------
    up_down_times : dictionary
        a dictionary of lists, of up and down timing in seconds
        keyed by leg, e.g. leg_dict['R4']['u']  ( = [ 2,5,6,8 ... ] )
        values are in seconds!
    latest_event : integer
        the time in milliseconds at which the last up or down step was recorded.

    '''
    up_down_times = {}
    latest_event = 0

    for leg_state in mov_data.keys():
        leg,state = leg_state.split('_')
        if leg not in up_down_times.keys():
            up_down_times[leg] = {}
        times = [float(x) for x in mov_data[leg_state].split()]
        if max(times) > latest_event:
            latest_event = max(times)
        if state == 'down':
            up_down_times[leg]['d'] = times
        elif state == 'up':
            up_down_times[leg]['u'] = times

    return up_down_times, latest_event

# quality control ... make sure up and down times are alternating!
def qcDownsUps(downs, ups):
    combo_times = np.array(downs + ups)
    
    down_array = ['d'] * len(downs)
    up_array = ['u'] * len(ups)
    combo_ud = np.array(down_array + up_array)
    inds = combo_times.argsort()
    sorted_ud = combo_ud[inds]
    sorted_times = combo_times[inds]
   
    problem = ''
    
    for i in range(len(sorted_ud[:-1])):
        
        current_state = sorted_ud[i]

        next_state = sorted_ud[i + 1]
        
        if current_state == next_state:
            problem += '\nalternating u/d problem!\n'
            problem += current_state + ' at ' + str(sorted_times[i]) + '\n'
            problem += next_state + ' at ' + str(sorted_times[i+1]) + '\n'
            
    return problem

# check for excel file for a movie clip
def check_for_excel(mov_file):
    file_stem = mov_file.split('.')[0]
    excel_filename = file_stem + '.xlsx'
    glob_list = glob.glob(excel_filename)
    if len(glob_list) > 0:
        excel_file_exists = True
    else:
        excel_file_exists = False
    return excel_file_exists, excel_filename

# quality control for leg_dict 
def qcLegDict(up_down_times):
    for leg in up_down_times.keys():
        downs = up_down_times[leg]['d']
        ups = up_down_times[leg]['u']
        problem = qcDownsUps(downs,ups)
        if len(problem) > 0:
            print('Problem for ' + leg)
            print(problem)

def listDirectories():
    dirs = next(os.walk(os.getcwd()))[1]
    dirs = sorted([d for d in dirs if d.startswith('_') == False and d.startswith('.') == False])
    return dirs

def select_movie_file():
    movie_files = sorted(glob.glob('*.mov'))
    if len(movie_files) > 0:
        movie_file = selectOneFromList(movie_files)
    else:
        movie_file = ''
        print('Cannot find a movie (.mov) file - do you have one here?')
    return movie_file

def identity_print_order():
    return ['file_stem','date','treatment','individualID','time_range',
            'initials','#frames','fps','width','height','duration']

def selectOneFromList(li):
    print('\nChoose from this list : ')
    i = 1
    li = sorted(li)
    
    for thing in li:
        print(str(i) + ': ' + thing)
        i += 1
    entry = input('\nWhich ONE do you want? ')
    choice = int(entry)
    
    if choice > len(li):
        err_message = '==> ' +  str(choice) + ' is invalid for list of ' + str(len(li)) + ' items!'
        exit(err_message)
    
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



def getVideoData(movie_file, printOut = True):
    if len(glob.glob(movie_file)) == 0:
        exit('Cannot find ' + movie_file)
    else:
        vid = cv2.VideoCapture(movie_file)
        vid_width  = int(vid.get(3))
        vid_height = int(vid.get(4))
        vid_fps = int(np.round(vid.get(5)))
        vid_frames = int(vid.get(7))
        vid.release()
        vid_length = np.around(vid_frames / vid_fps, decimals = 2)
    if printOut == True:
        printString = 'width: ' + str(vid_width)
        printString += ', height: ' + str(vid_height)
        printString += ', fps: ' + str(vid_fps)
        printString += ', #frames: ' + str(vid_frames)
        printString += ', duration: ' + str(vid_length)
        print(printString)
    return (vid_width, vid_height, vid_fps, vid_frames, vid_length)

def stanceSwingColors():
    stance_color = [0.95, 0.95, 0.95]
    swing_color = [0.15, 0.15, 0.15]
    return stance_color, swing_color

# addLegToPlot is deprecated in favor of plotStepsForLegs
# def addLegToPlot(f, a, ylev, footdown, footup, videoEnd=6.2):
#     steps = []
#     stepTimes = [0]
#     stance_color, swing_color = stanceSwingColors()

#     if footdown[0] < footup[0]:
#         steps.append('u')
#     else:
#         steps.append('d')

#     while len(footdown) > 0 and len(footup) > 0:
#         if footdown[0] < footup[0]:
#             steps.append('d')
#             stepTimes.append(footdown[0])
#             footdown = footdown[1:]
#         else:
#             steps.append('u')
#             stepTimes.append(footup[0])
#             footup = footup[1:]

#     # deal with last step
#     if len(footdown) > 0:
#         steps.append('d')
#         stepTimes.append(footdown[0])
#     elif len(footup) > 0:
#         steps.append('u')
#         stepTimes.append(footup[0])

#     lastStepTime = videoEnd
#     stepTimes.append(lastStepTime)
#     rectHeight = 1

#     for i, step in enumerate(stepTimes[:-1]):
#         if steps[i] == 'd':
#             fc = stance_color
#             ec = 'k'
#         else:
#             fc = swing_color
#             ec = 'k'

#         # ax.add_patch(Rectangle((1, 1), 2, 6))
#         a.add_patch(Rectangle((stepTimes[i], ylev - rectHeight / 2), stepTimes[i + 1] - stepTimes[i], rectHeight,
#                               edgecolor=None, facecolor=fc, fill=True, lw=1))

#     return f, a


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


def proportionsFromList(li):
    '''
    Parameters
    ----------
    li : list of strings
        a list containing a set of strings.

    Returns
    -------
    dict_with_proportional_values : dictionary
        keys = items in the input list
        values = proportion of input list that is each item.

    '''
    dict_with_numerical_values = {}
    dict_with_proportional_values = {}
    
    # count the number of each item in the input list
    for k in li:
        if k in dict_with_numerical_values.keys():
            dict_with_numerical_values[k] += 1
        else:
            dict_with_numerical_values[k] = 1
    
    # calculate proportions of each count
    for k in dict_with_numerical_values.keys():
        dict_with_proportional_values[k] = dict_with_numerical_values[k] / len(li)
        
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

def gaitStyleProportionsPlot(ax, movie_files, leg_set = 'lateral'):
    '''
 
    Parameters
    ----------
    ax : matplotlib axis object
        empty axis object.
    gait_style_vectors : list of vectors of gait_styles
        From gait_styles sheet in the experiment excel file.
        Can do more than one on a single plot
    leg_set : string, optional
        Which leg set to use. The default is 'lateral', but 'rear' is an option

    Returns
    -------
    ax : matplotlib axis object
        now filled with the gait style plot!

    '''
    
    barWidth = 0.1

    # get the gait vectors for the selected movie fiels
    gait_style_vectors = []
    exp_names = []
    
    for movie_file in movie_files:
        exp_name = movie_file.split('.')[0]
        exp_names.append(exp_name)
        times, gait_style_vector = getGaitStyleVec(movie_file, leg_set)
        gait_style_vectors.append(gait_style_vector)

    # set up colors
    if leg_set == 'rear':
        all_combos, combo_colors = get_gait_combo_colors('rear')
    else:
        all_combos, combo_colors = get_gait_combo_colors('lateral')

    exp_names = []
    for i, gait_styles_vec in enumerate(gait_style_vectors):
        
        combo_proportions = proportionsFromList(gait_styles_vec)

        for j, combo in enumerate(all_combos):

            if j == 0:
                bottom = 0

            if i == 0: # first dataset ... plot everything at 0 value to make labels for legend
                ax.bar(i, 0, bottom = bottom, color = combo_colors[combo],
                           edgecolor='white', width=barWidth, label=combo.replace('_',' '))

            if combo in gait_styles_vec:

                ax.bar(i, combo_proportions[combo], bottom = bottom, color = combo_colors[combo],
                    edgecolor='white', width=barWidth)

                bottom += combo_proportions[combo]

    ax.set_xticks(np.arange(len(gait_style_vectors)))
    if len(exp_names) > 1:
        ax.set_xticklabels(exp_names)
    else:
        ax.set_xticks([])

    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='upper left',
              bbox_to_anchor=(1,1), ncol=1, fontsize=8)

    ax.set_ylabel(leg_set + ' gait styles')
    ax.set_ylim([0,1])
    
    return ax

def need_tracking():
    sys.exit('\n ==> Need to run trackCritter.py before analyzing the path!\n')

def loadMovData(movie_file):
    '''
    get step up and down times for a movie
    
    Parameters
    ----------
    movie_file : string
        file name of a movie (.mov).

    Returns
    -------
    mov_data : dictionary
        keys = leg_state (e.g. 'L1_up', or 'R4_down')
        values = a string of integers representing timing in milliseconds
    excel_filename : string
        fine name of excel spreadsheet associated with movie_file.

    '''
    # 
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    if excel_file_exists:
        
        try:
            df = pd.read_excel(excel_filename, sheet_name='steptracking', index_col=None)
        except:
            needFrameStepper()
        
        try:
            mov_data = dict(zip(df['leg_state'].values, df['times'].values))
        except:
            needFrameStepper()

        if len(mov_data) < 16:
            exit('Need to finish tracking all legs with frameStepper.py! \n')
    else:
        import initializeClip
        initializeClip.main(movie_file)
        needFrameStepper()
    return mov_data, excel_filename

def loadStepData(movie_file):
    '''

    Parameters
    ----------
    movie_file : string
        file name of a movie (.mov).

    Returns
    -------
    step_data : pandas dataframe
        parameters for every step ... in step_timing sheet, from analyzeSteps.

    '''
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    
    if excel_file_exists:
        # check if data in step_timing sheet
        try:
            # if yes, load it as step_data_df
            step_data = pd.read_excel(excel_filename, sheet_name='step_timing', index_col=None)
        except:
            # if no, run analyzeSteps and get step_data_df
            import analyzeSteps
            step_data = analyzeSteps.main(movie_file)
    return step_data

def loadTrackedPath(movie_file):
    '''
    From pathtracking tab, produced by trackCritter (and analyzePath)
    frametimes, coordinates, per-frame speed, distance, bearings . . . 

    Parameters
    ----------
    movie_file : string
        file name of a movie (.mov).

    Returns
    -------
    tracked_data : pandas dataframe
        frametimes, coordinates, etc.
    excel_filename : string
        file name of excel file associated with the movie

    '''
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    
    if excel_file_exists:
    
        # load the tracked data from trackCritter
        tracked_data = pd.read_excel(excel_filename, sheet_name = 'pathtracking')
        if len(tracked_data) == 0:
            need_tracking()
    return tracked_data, excel_filename

def loadIdentityInfo(movie_file):
    '''
    get information about the clip from the identity tab of the excel spreadsheet associated with the clip

    Parameters
    ----------
    movie_file : string
        file name of a movie (.mov).

    Returns
    -------
    identity_info : dictionary
        movie identity information in identity sheet, from initializeClip
        keys = movie data parameters, e.g. date, treatment, individualID, time_range ...
        values = values for these parameters

    '''
    
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    if excel_file_exists:
        # check if data in the identity sheet
        try:
            identity_df = pd.read_excel(excel_filename, sheet_name='identity', index_col=None)
        except:
            exit('No data in identity sheet for ' + excel_filename)
    
        identity_info = dict(zip(identity_df['Parameter'].values, identity_df['Value'].values))
    else:
        exit('No excel file for this clip - run initializeClip!')
        
    return identity_info


def getFirstLastFrames(movie_file):
    filestem = movie_file.split('.')[0]
    
    first_frame_file = filestem + '_first.png'
    last_frame_file = filestem + '_last.png'
    
    if len(glob.glob(first_frame_file)) > 0:
        first_frame = cv2.imread(first_frame_file)
    else:
        print('... getting first frame ...')
        vidcap = cv2.VideoCapture(filestem + '.mov')
        success, image = vidcap.read()
        if success:
            first_frame = image
        else:
            print('cannot get an image from ' + filestem)
            first_frame = None
    
    if len(glob.glob(last_frame_file)) > 0:
        last_frame = cv2.imread(last_frame_file)
    else:
        print('... getting last frame ...')
        vidcap = cv2.VideoCapture(filestem + '.mov')
        frame_num = 1
        good_frame = None
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret == False:
                print('Last successful frame = ' + str(frame_num))
                last_frame = good_frame
                vidcap.release()
            else:
                frame_num += 1
                good_frame = frame
    
    return first_frame, last_frame

def saveFirstLastFrames(movie_file, first_frame, last_frame):
    cv2.imwrite(movie_file.split('.')[0] + '_first.png', first_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(movie_file.split('.')[0] + '_last.png', last_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def getFrameTimes(movie_file):
    
    needFrames = False
    
    # see if we can get the frame times from the excel file associated with this movie
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    if excel_file_exists:
        # see if we can read the data in the pathtracking sheet
        try:
            pathtracking_df = pd.read_excel(excel_filename, sheet_name='pathtracking', index_col=None)
        except:
            needFrames = True
            
        try:
            frame_times = pathtracking_df['times'].values
        except:
            needFrames = True
            
    else:
        needFrames = True
    
    # if needFrames is True, we need to run through the video, get the frames,
    # and save themn to the pathtracking sheet
    
    if needFrames:
        pathtracking_d = {}
        vid = cv2.VideoCapture(movie_file)
        fps = vid.get(5)
        frame_times = []
        frame_number = 0
        print('Getting frame times for ' + movie_file)
        while vid.isOpened():
            
            ret, frame = vid.read()
            
            if ret != True:  # no frame!
                print('... video end!')
                break
            
            frame_number += 1
            frameTime = round(float(frame_number)/fps,4)
            frame_times.append(frameTime)
            
        frame_times = np.array(frame_times)
        pathtracking_d['times'] = frame_times

        pathtracking_df = pd.DataFrame(pathtracking_d)
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            pathtracking_df.to_excel(writer, index=False, sheet_name='pathtracking')
    
    else:
        # if needFrames is False, we can get the frames from the pathtracking_df
        frame_times = pathtracking_df['times'].values
    
    return frame_times
        

def getGaits(movie_file, leg_set = 'lateral'):
    '''

    Parameters
    ----------
    movie_file: string
        file name of a movie of a walking tardigrade
        needs to be tracked with frameStepper before running this!
    leg_set : string
        either 'lateral' to get first 3 pairs, or 'rear' to get last pair

    Returns
    -------
    frame_times : numpy array
        frame times of movie clip, in seconds
    gait_styles : list
        list of gait styles, for every frame in frame_times
    up_legs : list
        list of which legs are swinging, for every frame in frame_times
    leg_matrix : numpy array
        rows = vector of swings (1's) and stances (0's) for each leg
        columns = each frame of video clip.

    '''
        
    # ===> See if we already have gait info in the excel_file
    # in the 'gait_styles' tab
    # get or make excel file for this clip
    need_gait_styles = False
    excel_file_exists, excel_filename = check_for_excel(movie_file)
    if excel_file_exists:
        
        try:
            gait_data = pd.read_excel(excel_filename, sheet_name='gait_styles')
        except:
            gait_data = pd.DataFrame({})
            
        if len(gait_data) > 0:
            # if we have that info, just return it!
            frame_times = gait_data.frametimes.values
            if leg_set == 'rear':
                gait_styles = gait_data.gaits_rear.values
                up_legs = gait_data.swinging_rear.values
            else:
                gait_styles = gait_data.gaits_lateral.values
                up_legs = gait_data.swinging_lateral.values
            # return frame_times, gait_styles, up_legs
        else:
            need_gait_styles = True
    
    else:
        import initializeClip
        initializeClip.main(movie_file)
        exit('Need to run frameStepper first!')
    
    # No gait data yet in excel file ... 
    # ... let's get it
    # ... and save it 
    
    if need_gait_styles:
        
        # Get frame times for this movie ... WITHOUT tracked path!
        frame_times = getFrameTimes(movie_file)
        
        # Get up_down_times for this movie
        mov_data, excel_filename = loadMovData(movie_file)
        up_down_times, latest_event = getUpDownTimes(mov_data)

        # trim frame_times to only include frames up to last recorded event
        last_event_frame = np.min(np.where(frame_times >= latest_event))
        frame_times_with_events = frame_times[:last_event_frame]
        
        if leg_set == 'rear':
            legs = get_leg_combos()['legs_4']
            all_combos, combo_colors = get_gait_combo_colors('rear')
        else:
            legs = get_leg_combos()['legs_lateral']
            all_combos, combo_colors = get_gait_combo_colors('lateral')
        
        # get leg matrix
        leg_matrix = make_leg_matrix(legs, up_down_times, frame_times_with_events)
        legs = np.array(legs)
        
        gait_styles = []
        up_legs = []
    
        for col_ind in np.arange(np.shape(leg_matrix)[1]):
            one_indices = np.where(leg_matrix[:, col_ind] == 1)
            swinging_legs = legs[one_indices]
            swinging_leg_combo = '_'.join(sorted(swinging_legs))
            up_legs.append(swinging_leg_combo)
            gait_styles.append(get_swing_categories(swinging_leg_combo, leg_set))
            
        # append the last swinging_leg_combo and gait_style to make the size same as frame_times
        extra_frames = len(frame_times)-len(gait_styles)
        gait_styles.extend([gait_styles[-1]] * extra_frames)
        up_legs.extend([up_legs[-1]] * extra_frames)
    
    return frame_times, gait_styles, up_legs

def saveGaits(movie_file):
    '''
    Save gait styles for lateral legs and rear legs ...
    ...to gait_styles of the excel spreadsheet associated with movie_file

    Parameters
    ----------
    movie_file: string
        file name of a movie of a walking tardigrade
        needs to be tracked with frameStepper before running this!

    Returns
    -------
    None.

    '''
    
    frame_times, lateral_gait_styles, lateral_up_legs = getGaits(movie_file, 'lateral')
    frame_times, rear_gait_styles, rear_up_legs = getGaits(movie_file, 'rear')
    
    d = {'frametimes':frame_times, 
         'gaits_lateral':lateral_gait_styles, 'swinging_lateral':lateral_up_legs, 
         'gaits_rear':rear_gait_styles, 'swinging_rear':rear_up_legs}
    
    df = pd.DataFrame(d)
    excel_filename = movie_file.split('.')[0] + '.xlsx'
    
    print('Saving gaits to gait_styles sheet ... ')
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='gait_styles')




def frameSwings(movie_file):
    '''
    From gait_style sheet in the excel file associated with movie_file
    ... make a dictionary = frames_swinging

    Parameters
    ----------
    movie_file : string
        filename of a movie clip

    Returns
    -------
    frames_swinging : dictionary
        keys = frame times in seconds.
        values = list of legs that are swinging during this frame

    '''

    frames_swinging = {}
    
    excel_file = movie_file.split('.')[0] + '.xlsx'
    
    try:
        gait_df = pd.read_excel(excel_file, sheet_name='gait_styles')
    except:
        print('No gait_styles sheet in ' + excel_file)
    
    frametimes = gait_df['frametimes'].values
    swinging_lateral = gait_df['swinging_lateral'].values
    swinging_rear = gait_df['swinging_rear'].values
    
    for i, frame in enumerate(frametimes):
        try:
            lateral_swings = swinging_lateral[i].split('_')
        except:
            lateral_swings = []
        try:
            rear_swings = swinging_rear[i].split('_')
        except:
            rear_swings = []
        frames_swinging[frame] = lateral_swings + rear_swings
        
    return frames_swinging


def plotLegSet(ax, movie_file, legs_to_plot = 'all'):  
    '''
    For one clip: step plots of given leg set
    
    Parameters
    ----------
    ax : matplotlib axis object
    
    movie_file : string
        file name of a movie (.mov).
    
    legs_to_plot : list (or the string 'all')
        which list of legs do we want to look at ... or 'all'
        
    Returns
    -------
    ax : matplotlib axis object
        now containing plotted steps for list in legs_to_plot (or all legs)
    '''
    
    frames_swinging = frameSwings(movie_file)
    
    # plot selected legs in same order as in leg_combos['legs_all']
    all_legs = get_leg_combos()['legs_all']
    
    if legs_to_plot == 'all':
        legs_to_plot = all_legs
    else:
        legs_to_plot = [x for x in all_legs if x in legs_to_plot]
    
    # plot with left on the top, right on the bottom    
    legs_to_plot = list(reversed(legs_to_plot))
        
    stance_color, swing_color = stanceSwingColors()
    
    frame_times = sorted(frames_swinging.keys())
    
    for i, leg in enumerate(legs_to_plot):
        for j, frame_time in enumerate(frame_times[:-1]):
            bar_width = frame_times[i+1] - frame_times[i]
            if leg in frames_swinging[frame_time]:
                bar_color = swing_color
            else:
                bar_color = stance_color
            ax.barh(i+1, bar_width, height=1, left = j*bar_width,
                       color = bar_color)

    ax.set_ylim([0.5, len(legs_to_plot)+0.5])
    ax.set_xlabel('Time (sec)')
    ax.set_yticks(np.arange(len(legs_to_plot))+1)
    ax.set_yticklabels(legs_to_plot)
    ax.set_ylabel('legs')
    ax.set_frame_on(False)
    
    return ax


def getGaitStyleVec(movie_file, leg_set = 'lateral'):
    excel_file = movie_file.split('.')[0] + '.xlsx'
    
    try:
        gait_df = pd.read_excel(excel_file, sheet_name='gait_styles')
    except:
        print('No gait_styles sheet in ' + excel_file)
        return None, None
    
    # get gait categories and colors for these categories        
    if leg_set == 'rear':
        all_combos, combo_colors = get_gait_combo_colors('rear')
        data_column = 'gaits_rear'
    else:
        all_combos, combo_colors = get_gait_combo_colors('lateral')
        data_column = 'gaits_lateral'
        
    times = gait_df.frametimes.values
    gait_styles = gait_df[data_column].values
    
    return times, gait_styles

def plotGaits(gaits_ax, movie_file, leg_set='lateral'):
    
    '''
    for ONE clip - plot steps with color-coded gait styles
    
    Parameters
    ----------
    gaits_ax : a matplotlib axis object
    
    movie_file : string
        file name of a movie (.mov).
    
    leg_set : string
        which set of legs do we want to look at?
        typically 'rear' or 'lateral'
    '''

    # get the gait styles info from the movie_file
    times, gait_styles = getGaitStyleVec(movie_file, leg_set)
    
    # get the plot colors for each gait style
    all_combos, combo_colors = get_gait_combo_colors(leg_set)

    previous_time = 0
    for i, style in enumerate(gait_styles):
        bar_width = times[i] - previous_time
        gaits_ax.barh(1, bar_width, height=0.6, left=previous_time, color = combo_colors[style])
        previous_time = times[i]

    gaits_ax.set_ylabel('gait\n' + leg_set)

    gaits_ax.set_yticks([])
    gaits_ax.set_xticks([])
    gaits_ax.set_frame_on(False)
    
    return gaits_ax #  = a matplotlib axis

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

# def get_leg_swing_combos(movie_folder, leg_set = 'lateral'): # lateral or rear

#     leg_combos = get_leg_combos()
#     if leg_set == 'rear':
#         legs = leg_combos['legs_4']
#     else:
#         legs = leg_combos['legs_lateral']
    
#     # set up variables to collect the data we need
#     leg_swing_combos = {}
#     total_frames = 0

#     # get frame_times for this movie (in milliseconds, e.g. [0 33 66 100 .... ])
#     frame_times = get_frame_times(movie_folder)
#     total_frames += len(frame_times)

#     # get dictionary of up & down timing for this video clip
#     # keys = leg['u'] or leg['d'] where leg is in ['L4','L3','L2','L1' (or rights)]
#     up_down_times, latest_event = getUpDownTimes(os.path.join(movie_folder, 'mov_data.txt'))

#     # get matrix of up (1's) and down (0's) data for all legs
#     # rows = legs
#     # columns = frames of video
#     leg_matrix = make_leg_matrix(legs, up_down_times, frame_times)

#     # get dictionary of #frames swinging for different combinations of legs 
#     leg_swing_counts = get_leg_swing_counts(leg_matrix, leg_set)

#     # get counts of #frames in each type of swing category
#     for combo in leg_swing_counts.keys():
#         swing_category = get_swing_categories(combo, leg_set)
#         if swing_category in leg_swing_combos.keys():
#             leg_swing_combos[swing_category] += leg_swing_counts[combo]
#         else:       
#             leg_swing_combos[swing_category] = leg_swing_counts[combo]
        
#     return leg_swing_combos

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


# def get_frame_times(movie_folder):
#     video_file = getMovieFromFileList(movie_folder)
#     vid = cv2.VideoCapture(os.path.join(movie_folder, video_file))
#     vidlength, numframes, vidfps, vidstart, frame_width, frame_height = getVideoStats(vid, False)
#     vid.release()
#     frame_times = np.array([int(x) for x in np.linspace(0, vidlength * 1000, int(numframes))])
#     return frame_times


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
    ''' 

    Parameters
    ----------
    legs : list or numpy array
        a list of leg names (e.g. ['R1','L1','R2','L2',... ])
        the order of this list will be the order of rows in the output matrix
    up_down_times : dictionary
        from getUpDownTimes here
        timing is seconds!
    frame_times : numpy array
        1 dimensional vector of all frame times from a movie ... in seconds!

    Returns
    -------
    leg_matrix : numpy array
        rows = vector of swings (1's) and stances (0's) for each leg
        columns = each frame of video clip.

    '''

    # make empty matrix
    leg_matrix = np.zeros([len(legs), len(frame_times)])

    # fill up each row with leg data
    for i, leg in enumerate(legs):
        # print(leg)
        ups = np.array(up_down_times[leg]['u'])
        downs = np.array(up_down_times[leg]['d'])
        leg_vector = up_down_times_to_binary(downs, ups, frame_times)
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
def removeFramesFolder(movie_file):
    frames_folder = movie_file.split('.')[0] + '_frames'
    fileList = glob.glob(frames_folder)
    if frames_folder in fileList:
        selection = input('Remove frames folder? (y) or (n): ')
        if selection == 'y':
            print(' ... removing ' + frames_folder + '\n')
            shutil.rmtree(frames_folder)

def cleanUpTrash(movie_file):
    fstem = movie_file.split('.')[0]
    bg = glob.glob(fstem + '*background.png')
    fi = glob.glob(fstem + '*first.png')
    la = glob.glob(fstem + '*last.png')
    for f in [bg, fi, la]:
        if len(f) > 0:
            os.remove(f[0])

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

def get_paired_step_parameters(parameters, legs, groups, dataframes, size_speed_dictionaries):
       
    print('Comparing parameters for ' + ', '.join(legs) + ' in ' + ' & '.join(groups))    
    print(' ... looking at ' + ', '.join(parameters))
    
    # get individuals from clip names
    clips = np.unique(dataframes[0]['clip'])
    tardigrades = np.unique(sorted([int(individualFromClipname(clipname)) for clipname in clips]))
    
    # dictionary of different types of data for different tardigrades
    # tardigrade_data['group']['tardigrade']['datatype'] = data
    tardigrade_data = {} # keyed on group
    
    for group in groups:
        
        if group not in tardigrade_data.keys():
            tardigrade_data[group] = {} # keyed on tardigrade
        
        for tardigrade in tardigrades:
            tardigrade_data[group][tardigrade] = {} # keyed on data type
    
    # go through speed / size and get the things
    
    # go through each parameter that is in the dataframe in get info for each clip and tardigrade
    for parameter in parameters:
    
        # go through each group   
        for i, group in enumerate(groups):

            # which dataframe are we looking at?
            df = dataframes[i]

            # which size & speed dictionary do we need?
            size_speed = size_speed_dictionaries[i]

            # get clips in df
            clips = np.unique(df['clip'])

            # go through each clip
            for clip in clips:

                # get individual from clip name
                tardigrade = int(individualFromClipname(clip))

                # get data from size_speed
                if parameter in size_speed[clip].keys():
                    # some of this data does not exist
                    # for example, some of the tardigrades do not have speed
                    data_to_plot = size_speed[clip][parameter]
                    
                elif parameter == 'gait_efficiency':
                    # gait efficiency = distance traveled per step
                    # if speed = distance / time ... and step is time / stance (i.e. stance_time)
                    # Then efficiency (distance / step) = speed * stance_time
                    speed = size_speed[clip]['tardigrade_speed']
                    data_from_clip = df[df['clip'] == clip]
                    gait_cycle = data_from_clip[data_from_clip['ref_leg'].isin(legs)]['stance_time']
                    try: # some tardigrades lack measurements for speed
                        data_to_plot = np.mean(speed) * np.mean(gait_cycle)
                    except: 
                        print('No speed for ' + clip + ', so no gait efficiency')
                        data_to_plot = False
                    
                else:
                    # get data from dataframe
                    data_from_clip = df[df['clip'] == clip]
                    data_to_plot = data_from_clip[data_from_clip['ref_leg'].isin(legs)][parameter]
                
                # need to check if there's any data before adding it to dictionary
                try:
                    mean_data_to_plot = np.mean(data_to_plot)
                except:
                    print('no ' + parameter + ' for ' + clip)
                else:
                    # add to tardigrade_data for this indvidual
                    if parameter in tardigrade_data[group][tardigrade].keys():
                        tardigrade_data[group][tardigrade][parameter].append(mean_data_to_plot)
                    else:
                        tardigrade_data[group][tardigrade][parameter] = [mean_data_to_plot]
        
    # done with groups ... return tardigrade data
    return tardigrade_data

def paired_comparison_plot(parameters, paired_data_for_parameters, groups=[]):
     
    # set up a figure
    num_parameters = len(parameters)
    
    max_plots_in_row = 4
    if num_parameters % max_plots_in_row == 0:
        num_rows = int(num_parameters / max_plots_in_row)
        num_cols = max_plots_in_row
    else: 
        num_rows = int(num_parameters/max_plots_in_row) + 1
        num_cols = max_plots_in_row     
    
    f,axes = plt.subplots(num_rows, num_cols, figsize = (14,3 * num_rows), constrained_layout = True)
    plot_colors = get_plot_colors()
    ms = 50 # marker size
    fs = 14 # fontsize
    mean_linewidth = 5
    mean_line_offset = 0.05
    connector_linewidth = 2
    
    if len(groups) == 0:
        groups = sorted(paired_data_for_parameters.keys())
    elif len(groups) > 2:
        print('Can only compare 2 groups!')
        return 
    else:
        for group in groups:
            if group not in paired_data_for_parameters.keys():
                print('No ' + group + ' in paired_data_for_parameters!')
                return
    
    num_tardigrades = len(paired_data_for_parameters[groups[0]].keys())
    connector_xcoords = np.zeros((num_tardigrades,2))
    connector_xcoords[:,1] = 1
    
    for p, parameter in enumerate(parameters):
        
        connector_ycoords = np.zeros((num_tardigrades,2))
        
        # go through each group
        for g, group in enumerate(groups):                
            
            # go through each tardigrade
            for t, tardigrade in enumerate(paired_data_for_parameters[group].keys()):
                
                # scatter plot of values for each tardigrade, with appropriate color
                y_data = paired_data_for_parameters[group][tardigrade][parameter]
                x_data = [g] * len(y_data)
                f.axes[p].scatter(x_data, y_data, s = ms, c = plot_colors[tardigrade], alpha = 0.1)
                
                
                # horizontal line at mean value of data
                mean_val = np.mean(y_data)
                
                # homemade error bar . . . 
                y_err = sem(y_data)
                f.axes[p].plot([g,g],[mean_val-y_err, mean_val+y_err],
                              linewidth = mean_linewidth-1, color = plot_colors[tardigrade])
                
                if p == 0 and g == 0:
                    f.axes[p].plot([g-mean_line_offset, g+mean_line_offset], [mean_val, mean_val],
                            linewidth = mean_linewidth, color = plot_colors[tardigrade],
                                   label = '# ' + str(tardigrade))
                else:
                    f.axes[p].plot([g-mean_line_offset, g+mean_line_offset], [mean_val, mean_val],
                            linewidth = mean_linewidth, color = plot_colors[tardigrade])
                
                # update coordinate for a line connecting means
                connector_ycoords[t][g] = mean_val
                
                # plot line connecting means for this tardigrade
                
            if p == 0:
                # add legend to axis
                f.axes[p].legend(fontsize = fs)
        
        # done with parameter, plot line connecting means for each tardigrade
        for r, ycoords in enumerate(connector_ycoords):
            col = plot_colors[r+1]
            f.axes[p].plot(connector_xcoords[r,:], ycoords, color = col, linewidth = connector_linewidth)
    
    for i, parameter in enumerate(parameters):
        f.axes[i].set_ylabel(parameters[i].replace('_',' '), fontsize = fs)
        f.axes[i].set_xlim([-0.5, 1.5])
        f.axes[i].set_xticks([0,1],[group.replace('_', ' ') for group in groups], fontsize = fs)
        
    if len(parameters) < len(f.axes):
        empty_axes = f.axes[len(parameters):]
        for ax in empty_axes:
            ax.axis('off')
    
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
        plot_colors = np.array(['firebrick','gold','forestgreen', 'darkviolet', 'steelblue',
        'darkorange', 'lawngreen', 'gainsboro', 'black'])

    if num_colors > len(plot_colors):
        print('too many colors')
        return plot_colors
    else:
        return plot_colors[:num_colors]
    
def needFrameStepper():
    sys.exit('==> Need to track legs with frameStepper.py <== \n')
    
    
