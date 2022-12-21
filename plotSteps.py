#!/usr/bin/python
import gaitFunctions
import sys
import pandas as pd
# import glob
# import shutil
# import cv2
# import os

def main(movie_file):

    # check if excel file exists
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    if excel_file_exists:
        df = pd.read_excel(excel_filename, sheet_name='steptracking', index_col=None)
        
        try:
            mov_data = dict(zip(df['leg_state'].values, df['times'].values))
        except:
            gaitFunctions.needFrameStepper()

        if len(mov_data) < 16:
            exit('Need to finish tracking all legs with frameStepper.py! \n')
            
    else:
        # if no, run initializeClip.py and prompt to do frameStepper
        import initializeClip
        initializeClip.main(movie_file)
        gaitFunctions.needFrameStepper() # this exits

    # check if data in step_timing sheet
    try:
        # if yes, load it as step_data_df
        step_data_df = pd.read_excel(excel_filename, sheet_name='step_timing', index_col=None)
    except:
        # if no, run analyzeSteps and get step_data_df
        import analyzeSteps
        step_data_df = analyzeSteps.main(movie_file)

    # ==> get information about the clip from the identity tab

    # ==> ask to remove the frames folder if it exists
    gaitFunctions.removeFramesFolder(data_folder)

    # parse movie data to make a dictionary containing up and down timing for each leg
    # e.g. leg_dict['R4']['u']  ( = [ 2,5,6,8 ... ] )
    leg_dict, video_end = gaitFunctions.getUpDownTimes(mov_data) # <=== CHECK THIS ... update leg_dict to up_down_times?

    # quality control on leg_dict ... make sure up and down times are alternating!
    gaitFunctions.qcLegDict(leg_dict)

    # plot steps - choose which legs to plot
    legs = gaitFunctions.get_leg_combos()['legs_all']  # dictionary of all combos
    # OR choose individual legs to plot
    # legs = ['L4','R4'] # for an individual leg
    # plot_legs(leg_dict, legs, video_end)

    # save a bunch of figures of leg plots
    gaitFunctions.save_leg_figures(data_folder, leg_dict, video_end)

    # save stance time and swing time figures
    gaitFunctions.save_stance_figures(data_folder, leg_dict, legs)

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
