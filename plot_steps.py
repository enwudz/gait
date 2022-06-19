#!/usr/bin/python
from gait_analysis import *

# get data ... which folder should we look in?
# run this script in analyzed_movies
dirs = listDirectories()
data_folder = selectOneFromList(dirs)
mov_data = os.path.join(data_folder, 'mov_data.txt')

# parse movie data to get leg info
# make dictionaries for each leg for which we have data
leg_dict, video_end = getUpDownTimes(mov_data)

# quality control on leg_dict
qcUpDownTimes(leg_dict)

# plot steps - choose which legs to plot
legs = get_leg_combos()['legs_all']  # dictionary of all combos

# OR choose individual legs to plot
# legs = ['L4','R4'] # for an individual leg
# plot_legs(leg_dict, legs, video_end)

# save a bunch of figures of leg plots
save_leg_figures(data_folder, leg_dict, video_end)

# get (and plot?) stance length per leg, swing length per leg
stance_data = plot_stance(leg_dict, legs, 'stance', False)[0]
swing_data = plot_stance(leg_dict, legs, 'swing', False)[0]

# save stance and swing data figures
save_stance_figures(data_folder, leg_dict, legs)

# save stance and swing data to a file
save_stance_swing(data_folder, legs, stance_data, swing_data)

