#!/usr/bin/python
from gait_analysis import *
import sys
import glob
import shutil
import cv2

def main(data_folder):

    # get data ... which folder should we look in?
    # run this script in a directory that has directories containing data for clips
    if len(data_folder) == 0: 
        dirs = listDirectories()
        data_folder = selectOneFromList(dirs)
    mov_data = os.path.join(data_folder, 'mov_data.txt')
    fileTest(mov_data)

    # parse movie data to get info about movie 
    # (movie name, analyzed length, frame range for speed calculation, movie length)
    movie_info = getMovieInfo(data_folder)

    # if we have information about what frames to use to calculate speed
    # ... save these two frames if we do not already have them
    saveSpeedFrames(data_folder, movie_info)

    # remove the frames folder if it exists
    removeFramesFolder(data_folder)

    # parse movie data to make a dictionary containing up and down timing for each leg
    # e.g. leg_dict['R4']['u']  ( = [ 2,5,6,8 ... ] )
    leg_dict, video_end = getUpDownTimes(mov_data)

    # quality control on leg_dict ... make sure up and down times are alternating!
    qcUpDownTimes(leg_dict)

    # plot steps - choose which legs to plot
    legs = get_leg_combos()['legs_all']  # dictionary of all combos
    # OR choose individual legs to plot
    # legs = ['L4','R4'] # for an individual leg
    # plot_legs(leg_dict, legs, video_end)

    # save a bunch of figures of leg plots
    save_leg_figures(data_folder, leg_dict, video_end)

    # save stance time and swing time figures
    save_stance_figures(data_folder, leg_dict, legs)

    # get (and plot?) stance length per leg, swing length per leg
    # stance_data, f, a = plot_stance(leg_dict, legs, 'stance', False)
    # swing_data, f, a = plot_stance(leg_dict, legs, 'swing', False)
    # save stance and swing data to a file
    #save_stance_swing(data_folder, legs, stance_data, swing_data)

def saveSpeedFrames(data_folder, movie_info):
    
    # check to see if we have the beginning and ending frames to calculate speed
    haveSpeedFrames = False
    beginning_speed_frame = os.path.join(data_folder,'beginning_speed_frame.png')
    ending_speed_frame = os.path.join(data_folder,'ending_speed_frame.png')
    fileList = glob.glob(os.path.join(data_folder, '*'))
    if beginning_speed_frame in fileList and ending_speed_frame in fileList:
        print(' ... found the speed frames in ' + data_folder)

    elif movie_info['speed_start'] > 0 and movie_info['speed_end'] > 0:
        print(' ... could not find the speed frames, saving them!')

        beginning_speed_range = int(movie_info['speed_start'] * 1000)
        ending_speed_range = int(movie_info['speed_end'] * 1000)
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
            if line.startswith('Speed') and 'none' not in line:
                speedRange = dataAfterColon(line)
                movie_info['speed_framerange'] = speedRange
                movie_info['speed_start'], movie_info['speed_end'] = getRangeFromText(speedRange)

        # if no information for 'Analyzed Frames', retrieve it from the movie
        if movie_info['start_frame'] == 0 or movie_info['end_frame'] == 0:
            print('No info about analyzed frame times ... getting that now ... ')
            first_frame, last_frame = getStartEndTimesFromMovie(data_folder, movie_info)
            print('   ... start is ' + str(first_frame) + ', end is ' + str(last_frame))
            movie_info['start_frame'], movie_info['end_frame'] = first_frame, last_frame
            movie_info['analyzed_framerange'] = str(first_frame) + '-' + str(last_frame)

    return movie_info

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

if __name__== "__main__":
    if len(sys.argv) > 1:
            data_folder = sys.argv[1]
            print('looking in ' + data_folder)
    else:
        data_folder = ''

    main(data_folder)
