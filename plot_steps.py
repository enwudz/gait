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

    # update mov_data.txt
    updateMovieData(data_folder, movie_info)

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

    # run save_step_data.py = gets information about every step of every leg
    # including timing of other legs' swings 
    get_step_data = input('Run save_step_data.py? (y) or (n): ')
    if get_step_data == 'y':
        import save_step_data
        save_step_data.main(data_folder)
    else:
        print('No worries - can run save_step_data.py later! ')

if __name__== "__main__":
    if len(sys.argv) > 1:
            data_folder = sys.argv[1]
            print('looking in ' + data_folder)
    else:
        data_folder = ''

    main(data_folder)
