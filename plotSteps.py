#!/usr/bin/python
import gaitFunctions
import sys
import matplotlib.pyplot as plt
import pandas as pd

def main(movie_file):
    
    if len(movie_file) > 0:
        # just one clip
        num_groups = 1
        movie_files = [movie_file]
    
    else: 
        # no movie selected. 
        # prompt to select a GROUP or GROUPs of movies
        pass
    
    for movie_file in movie_files:
        print('Getting data for ' + movie_file)
        
        # get mov_data and excel_file for this movie_file
        mov_data, excel_filename = gaitFunctions.loadMovData(movie_file)

        # get step_data for this movie_file
        step_data = gaitFunctions.loadStepData(movie_file)

        # ==> get information about the clip from the identity tab
        identity_info = gaitFunctions.loadIdentityInfo(movie_file)
        
        # ask to remove the frames folder if it exists
        gaitFunctions.removeFramesFolder(movie_file)
        
        # save a bunch of different combos of steps for ONE clip
        # saveBunchOfPlots(mov_data) # <=== OK ... for ONE clip
        
        # plot of steps and gait styles for ONE clip
        fig = gaitFunctions.plotStepsAndGait(movie_file, 'lateral') # <=== OK ... for ONE clip
        ##### BUT SHOULD CHANGE BECAUSE CODE IS DUPLICATED FOR GETTING GAIT!
        plt.show()
        


def saveBunchOfPlots(mov_data):
    # parse movie data to make a dictionary containing up and down timing for each leg
    # e.g. leg_dict['R4']['u']  ( = [ 2,5,6,8 ... ] )
    up_down_times, video_end = gaitFunctions.getUpDownTimes(mov_data)

    # quality control on leg_dict ... make sure up and down times are alternating!
    gaitFunctions.qcLegDict(up_down_times)

    # plot steps - choose which legs to plot
    legs = gaitFunctions.get_leg_combos()['legs_all']  # dictionary of all combos
    # OR choose individual legs to plot
    # legs = ['L4','R4'] # for an individual leg
    # plot_legs(leg_dict, legs, video_end)

    # save a bunch of figures of leg plots
    gaitFunctions.save_leg_figures(movie_file, up_down_times, video_end)

    # save stance time and swing time figures
    gaitFunctions.save_stance_figures(movie_file, up_down_times, legs)

# recreate plots from step_data_plots.ipynb

'''
How much data: # steps (each leg, and total), total time
    input is step_data   
'''
#

'''
Step plots of given leg set
    function is gaitFunctions.plotStepsAndGait
'''
#

'''
Compare gait parameters for different legs
    4 subplots: stance, swing, gait, duty
    for all legs (or a subset)
    (from step_data)
'''
#

'''
Compare gait parameters across body (left vs. right lateral legs)
    4 subplots: stance, swing, gait, duty
    (from step_data)
'''
#

'''
Compare metachronal lag across body (left vs. right lateral legs)
    2 subplots: raw vs. normalized
    (from step_data)
'''
#

'''
scatter speed vs. gait parameters for lateral legs
    gait parameters = stance, swing, gait, duty

'''
#

'''
scatter size vs. speed for all tardigrades
'''

'''
average step pattern for experiment (or group of experiments)
'''
#

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        
       movie_file = []

    main(movie_file)
