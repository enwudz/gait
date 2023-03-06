# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import os
# import shutil
# import glob
import pandas as pd
import gaitFunctions
import numpy as np

movie_file = 'kt_feb23_cat1_010-011.mov'

# load tracked path data
tracked_df, excel_filename = gaitFunctions.loadTrackedPath(movie_file)
# frametimes = tracked_df.times.values
bearings = tracked_df.bearings.values
print(np.std(bearings))


# turns = tracked_df.turns.values
# stops = tracked_df.stops.values



### batch rename files
# for filename in glob.glob('kt_*'):
#     new_filename = filename.replace('kt_','iw_')
#     shutil.move(filename, new_filename)

