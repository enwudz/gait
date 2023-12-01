#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:44:02 2023

@author: iwoods
"""

import pandas as pd
import numpy as np
import glob

def find_nearest(array, value):
    array = np.asarray(array)
    if len(array) > 0:
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx
    else:
        return 0, 0

excel_files = glob.glob('*.xlsx')

stems = []
clips = []
bouttiming = []
boutduration = []

# build up lists for stems, clips, bouttiming, and boutduration
for f in excel_files:
    pathstats_df = pd.read_excel(f, sheet_name='path_stats', index_col=None)
    cruise_bouts = pathstats_df[pathstats_df['path parameter'] == 'cruise bout timing'].values[0][1]
    cruise_durations = pathstats_df[pathstats_df['path parameter'] == 'cruise bout durations'].values[0][1]
    try:
        bouts = cruise_bouts.split(';')
        clip = f.split('.')[0]
        stem = '_'.join(clip.split('_')[:4])
        durations = cruise_durations.split(';')
        for i,bout in enumerate(bouts):
            stems.append(stem)
            clips.append(clip)
            bouttiming.append(bout)
            boutduration.append(float(durations[i]))
    except:
        None
  
# add each list into dictionary
d = {}
d['stems'] = stems
d['clips'] = clips
d['bout timing'] = bouttiming
d['bout duration'] = boutduration

df = pd.DataFrame(d)

# get unique movies
unique_movies = sorted(np.unique(df['stems'].values))

# for each movie, find the bout that is closest to the target length
# collect bouts until reach a threshold of total time
target = 10
threshold = 8 # in seconds (10? 8?)

for movie in unique_movies:
    cumulative_duration = 0
    movie_df = df[df['stems'] == movie]
    movie_df = movie_df.sort_values(by=['bout duration'], ascending=False)
    
    movie_durations = movie_df['bout duration'].values
    movie_bouttiming = movie_df['bout timing'].values
    movie_clips = movie_df['clips'].values

    while cumulative_duration < threshold:
        closest_clip, idx = find_nearest(movie_durations, target)
        if idx > 0:
            print(movie_clips[idx], movie_bouttiming[idx], movie_durations[idx])
            
            # remove items at this index
            movie_durations = np.delete(movie_durations,idx)
            movie_bouttiming = np.delete(movie_bouttiming, idx)
            movie_clips = np.delete(movie_clips, idx)
            
            cumulative_duration += closest_clip
        else:
            break
          
    # for i, movie_clip in enumerate(movie_clips):
    #     if cumulative_duration <= threshold:
    #         print(movie, movie_clip, movie_bouttiming[i], movie_durations[i])
    #         cumulative_duration += movie_durations[i]
    
    
    