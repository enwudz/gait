#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:01:45 2023

@author: iwoods

Combines data from all experiments in a folder into one excel spreadsheet

Each movie file should have an associated spreadsheet, which is produced by
    one or more of the following:
        trackCritter --> analyzeTrack
        frameStepper --> analyzeSteps

path_summaries = summary data from trackCritter and analyzeTrack
    each row is an individual
    each column is an average value across clips (mean or median)
step_timing = individual step data from frameStepper and analyzeSteps
    each row is a step
    each column is a measured parameter or other information about that step
step_summaries = summary data from step_timing
    each row is an individual
    each column is an average value of a step parameter across clips
gait_summaries = summary data from gait_styles
    each row is an individual
    each column is a gait style, and the values are the % time in that 
        gait style for that individual
        
Can use the resulting excel file to:
    compare treatments = average values for each individual in a group
    within an individual, see complete picture of gait style and step parameters
    
"""

import pandas as pd
import numpy as np
# import sys
import glob
import os
import gaitFunctions


def main():
    
    # set output file name to name of current directory
    current_directory = os.getcwd().split(os.sep)[-1]
    out_file = current_directory + '_combined.xlsx'
    
    # get list of excel files and make sure there is an excel for each movie
    clipstems = sorted(get_clips())

    # make empty dataframes to collect data
    path_summaries_df = pd.DataFrame()
    step_timing_combined_df = pd.DataFrame()
    step_summaries_df = pd.DataFrame()
    gait_summaries_df = pd.DataFrame()
    
    # make empty dictionaries to collect path data, keyed by unique individual
    clip_scales = {}
    clip_areas = {} 
    clip_lengths = {}
    clip_cruising = {}
    clip_duration = {}
    num_stops = {}
    num_turns = {}
    bearing_changes = {}
    distance_traveled = {}
    
    # make empty dictionaries to collect step parameters, keyed by unique individual
    clip_stance_lateral = {}
    clip_swing_lateral = {}
    clip_gait_lateral = {}
    clip_duty_lateral = {}
    
    clip_stance_rear = {}
    clip_swing_rear = {}
    clip_gait_rear = {}
    clip_duty_rear = {}
    
    clip_pixels_per_step_lateral = {}
    clip_pixels_per_step_rear = {}
    
    clip_anterior_offset = {}
    clip_opposite_offset_lateral = {}
    clip_opposite_offset_rear = {}
    
    clip_anterior_offset_normalized = {}
    clip_opposite_offset_lateral_normalized = {}
    clip_opposite_offset_rear_normalized = {}
    
    clip_metachronal_lag = {}
    clip_metachronal_lag_normalized = {}
    
    # make empty dictionaries to collect gait style times, keyed by unique individual
    clip_total_frames = {}
    clip_stand_lateral = {}
    clip_pentapod = {}
    clip_tetrapod_canonical = {}
    clip_tetrapod_gallop = {}
    clip_tetrapod_other = {}
    clip_tripod_canonical = {}
    clip_tripod_other = {}
    clip_other_lateral = {}
    
    clip_stand_rear = {}
    clip_hop = {}
    clip_step = {}
    
    
    # go through clips and collect data
    for clip in clipstems:
        movie_file = clip + '.mov'
        excel_file = clip + '.xlsx'
        print(' ... loading data for ' + movie_file)
        
        #### ===> load identity info from this clip
        identity_info = gaitFunctions.loadIdentityInfo(movie_file, excel_file)
        treatment = identity_info['treatment']
        individual = str(identity_info['individualID'])
        date = str(identity_info['date'])
        # print(treatment, individual, date)
        uniq_id = '_'.join([treatment, individual, date])
        
        #### ===> load path_stats from this clip
        path_stats_dict = gaitFunctions.loadPathStats(movie_file)
        
        # collect scale for this individual for this clip
        if uniq_id in clip_scales.keys():
            clip_scales[uniq_id] = np.append(clip_scales[uniq_id], float(path_stats_dict['scale']))
        else:
            clip_scales[uniq_id] = np.array(float(path_stats_dict['scale']))
        
        # collect areas for this individual in this clip
        if uniq_id in clip_areas.keys():
            clip_areas[uniq_id] = np.append(clip_areas[uniq_id], float(path_stats_dict['area']))
        else:
            clip_areas[uniq_id] = float(path_stats_dict['area'])
        
        # collect lengths for this individual in this clip
        if uniq_id in clip_lengths.keys():
            clip_lengths[uniq_id] = np.append(clip_lengths[uniq_id], float(path_stats_dict['length']))
        else:
            clip_lengths[uniq_id] = float(path_stats_dict['length'])
        
        # collect clip duration for this individual for this clip
        if uniq_id in clip_duration.keys():
            clip_duration[uniq_id] = np.append(clip_duration[uniq_id], float(path_stats_dict['clip duration']))
        else:
            clip_duration[uniq_id] = np.array(float(path_stats_dict['clip duration']))
                                              
        # collect #stops for this individual for this clip
        if uniq_id in num_stops.keys():
            num_stops[uniq_id] = np.append(num_stops[uniq_id], int(path_stats_dict['# stops']))
        else:
            num_stops[uniq_id] = np.array(int(path_stats_dict['# stops']))
        
        # collect #turns for this individual for this clip
        if uniq_id in num_turns.keys():
            num_turns[uniq_id] = np.append(num_turns[uniq_id], int(path_stats_dict['# turns']))
        else:
            num_turns[uniq_id] = np.array(int(path_stats_dict['# turns']))
        
        # collect bearing_changes for this individual for this clip
        if uniq_id in bearing_changes.keys():
            bearing_changes[uniq_id] = np.append(bearing_changes[uniq_id], float(path_stats_dict['cumulative bearings']))
        else:
            bearing_changes[uniq_id] = np.array(float(path_stats_dict['cumulative bearings']))
            
        # collect distance_traveled for this individual for this clip
        if uniq_id in distance_traveled.keys():
            distance_traveled[uniq_id] = np.append(distance_traveled[uniq_id], float(path_stats_dict['total distance']))
        else:
            distance_traveled[uniq_id] = np.array(float(path_stats_dict['total distance']))
        
        #### ===> load tracked data from this clip
        tdf, excel_file = gaitFunctions.loadTrackedPath(movie_file)
        if tdf is not None:
                     
            # collect cruising frames for this individual in this clip
            stop_frames = tdf['stops'].values
            turn_frames = tdf['turns'].values
            noncruise_frames = stop_frames + turn_frames
            cruise_frames = np.zeros(len(stop_frames))
            cruise_frames[np.where(noncruise_frames==0)] = 1
            if uniq_id in clip_cruising.keys():
                clip_cruising[uniq_id] = np.append(clip_cruising[uniq_id], cruise_frames)
            else:
                clip_cruising[uniq_id] = cruise_frames
        
        #### ===> load step_timing data from this clip
        sdf = gaitFunctions.loadStepData(movie_file, excel_file)
        
        # add step_timing data and identifying info to combined file
        if sdf is not None:
            sdf = addColtoDF(sdf, 'clip', clip) # add clip name
            sdf = addColtoDF(sdf, 'treatment', treatment) # add treatment type
            sdf = addColtoDF(sdf, 'individual', individual) # add individual name
            sdf = addColtoDF(sdf, 'date', date) # add date
            sdf = addColtoDF(sdf, 'uniq_id', treatment + '_' + individual + '_' + date)
            step_timing_combined_df = pd.concat([step_timing_combined_df, sdf])
            
        #### ===> make new summary sheet for step parameters = step_summaries

            lateral_legs = gaitFunctions.get_leg_combos()[0]['lateral']
            rear_legs = gaitFunctions.get_leg_combos()[0]['rear']
            
            # get steps while critter is 'cruising' 
            # (not turning, not stopping ... as determined by analyzePath.py)
            cruising = sdf[sdf['cruising_during_step'] == True]
            cruising_lateral = cruising[cruising['legID'].isin(lateral_legs)]
            cruising_rear = cruising[cruising['legID'].isin(rear_legs)]
            
            ## LATERAL step parameters
            # get stance duration for lateral legs for this individual for this clip
            if uniq_id in clip_stance_lateral.keys():
                clip_stance_lateral[uniq_id] = np.append(clip_stance_lateral[uniq_id], cruising_lateral['stance'].values)
            else:
                clip_stance_lateral[uniq_id] = cruising_lateral['stance'].values
                
            # get swing duration for lateral legs for this individual for this clip
            if uniq_id in clip_swing_lateral.keys():
                clip_swing_lateral[uniq_id] = np.append(clip_swing_lateral[uniq_id], cruising_lateral['swing'].values)
            else:
                clip_swing_lateral[uniq_id] = cruising_lateral['swing'].values
                
            # get gait cycle for lateral legs for this individual for this clip
            if uniq_id in clip_gait_lateral.keys():
                clip_gait_lateral[uniq_id] = np.append(clip_gait_lateral[uniq_id], cruising_lateral['gait'].values)
            else:
                clip_gait_lateral[uniq_id] = cruising_lateral['gait'].values
                
            # get duty factor for lateral legs for this individual for this clip
            if uniq_id in clip_duty_lateral.keys():
                clip_duty_lateral[uniq_id] = np.append(clip_duty_lateral[uniq_id], cruising_lateral['duty'].values)
            else:
                clip_duty_lateral[uniq_id] = cruising_lateral['duty'].values
        
            # get distance in pixels per lateral leg step (will convert later)
            if uniq_id in clip_pixels_per_step_lateral.keys():
                clip_pixels_per_step_lateral[uniq_id] = np.append(clip_pixels_per_step_lateral[uniq_id], cruising_lateral['distance_during_step'].values)
            else:
                clip_pixels_per_step_lateral[uniq_id] = cruising_lateral['distance_during_step'].values

            ## REAR step parameters
            # get stance duration for REAR legs for this individual for this clip
            if uniq_id in clip_stance_rear.keys():
                clip_stance_rear[uniq_id] = np.append(clip_stance_rear[uniq_id], cruising_rear['stance'].values)
            else:
                clip_stance_rear[uniq_id] = cruising_rear['stance'].values
                
            # get swing duration for rear legs for this individual for this clip
            if uniq_id in clip_swing_rear.keys():
                clip_swing_rear[uniq_id] = np.append(clip_swing_rear[uniq_id], cruising_rear['swing'].values)
            else:
                clip_swing_rear[uniq_id] = cruising_rear['swing'].values
                
            # get gait cycle for rear legs for this individual for this clip
            if uniq_id in clip_gait_rear.keys():
                clip_gait_rear[uniq_id] = np.append(clip_gait_rear[uniq_id], cruising_rear['gait'].values)
            else:
                clip_gait_rear[uniq_id] = cruising_rear['gait'].values
                
            # get duty factor for rear legs for this individual for this clip
            if uniq_id in clip_duty_rear.keys():
                clip_duty_rear[uniq_id] = np.append(clip_duty_rear[uniq_id], cruising_rear['duty'].values)
            else:
                clip_duty_rear[uniq_id] = cruising_rear['duty'].values
        
            # get distance in pixels per lateral leg step (will convert later)
            if uniq_id in clip_pixels_per_step_rear.keys():
                clip_pixels_per_step_rear[uniq_id] = np.append(clip_pixels_per_step_rear[uniq_id], cruising_rear['distance_during_step'].values)
            else:
                clip_pixels_per_step_rear[uniq_id] = cruising_rear['distance_during_step'].values    
        
            ## OFFSETS AND METACHRONAL LAG
            # get metachronal lag (lateral legs)
            left_metachronal_lag, right_metachronal_lag, mean_gait_cycle = gaitFunctions.getMetachronalLag(sdf)
            metachronal_lag = np.concatenate([left_metachronal_lag, right_metachronal_lag])
            # print(uniq_id, metachronal_lag) # just testing
            if uniq_id in clip_metachronal_lag.keys():
                clip_metachronal_lag[uniq_id] = np.append(clip_metachronal_lag[uniq_id], metachronal_lag)
            else:
                clip_metachronal_lag[uniq_id] = metachronal_lag / mean_gait_cycle
                
            # get metachronal lag normalized to gait cycle (lateral legs)
            if uniq_id in clip_metachronal_lag_normalized.keys():
                clip_metachronal_lag_normalized[uniq_id] = np.append(clip_metachronal_lag_normalized[uniq_id], metachronal_lag)
            else:
                clip_metachronal_lag_normalized[uniq_id] = metachronal_lag / mean_gait_cycle
            
            anterior_offsets, opposite_offsets_lateral, opposite_offsets_rear, mean_gait_cycle_lateral, mean_gait_cycle_rear = gaitFunctions.getSwingOffsets(sdf)
            # print(uniq_id, anterior_offsets) # just testing
            
            # get anterior swing offsets for lateral legs
            if uniq_id in clip_anterior_offset.keys():
                clip_anterior_offset[uniq_id] = np.append(clip_anterior_offset[uniq_id], anterior_offsets)
            else:
                clip_anterior_offset[uniq_id] = anterior_offsets
                
            # get anterior swing offsets normalized to gait cycle for lateral legs
            if uniq_id in clip_anterior_offset_normalized.keys():
                clip_anterior_offset_normalized[uniq_id] = np.append(clip_anterior_offset_normalized[uniq_id], anterior_offsets / mean_gait_cycle_lateral)
            else:
                clip_anterior_offset_normalized[uniq_id] = anterior_offsets / mean_gait_cycle_lateral
            
            # get opposite swing offsets for lateral legs
            if uniq_id in clip_opposite_offset_lateral.keys():
                clip_opposite_offset_lateral[uniq_id] = np.append(clip_opposite_offset_lateral[uniq_id], opposite_offsets_lateral)
            else:
                clip_opposite_offset_lateral[uniq_id] = opposite_offsets_lateral
            
            # get opposite swing offsets normalized to gait cycle for lateral legs
            if uniq_id in clip_opposite_offset_lateral_normalized.keys():
                clip_opposite_offset_lateral_normalized[uniq_id] = np.append(clip_opposite_offset_lateral_normalized[uniq_id], opposite_offsets_lateral / mean_gait_cycle_lateral)
            else:
                clip_opposite_offset_lateral_normalized[uniq_id] = opposite_offsets_lateral / mean_gait_cycle_lateral
            
            # get opposite swing offsets for rear legs
            if uniq_id in clip_opposite_offset_rear.keys():
                clip_opposite_offset_rear[uniq_id] = np.append(clip_opposite_offset_rear[uniq_id], opposite_offsets_rear)
            else:
                clip_opposite_offset_rear[uniq_id] = opposite_offsets_rear
                
            # get opposite swing offsets for rear legs
            if uniq_id in clip_opposite_offset_rear_normalized.keys():
                clip_opposite_offset_rear_normalized[uniq_id] = np.append(clip_opposite_offset_rear_normalized[uniq_id], opposite_offsets_rear / mean_gait_cycle_rear)
            else:
                clip_opposite_offset_rear_normalized[uniq_id] = opposite_offsets_rear / mean_gait_cycle_rear
            
            # left/right balance??
        
        #### ===> load gait_styles data from this clip
        gdf = gaitFunctions.loadGaitData(movie_file, excel_file)
        
        if gdf is not None:
            
            lateral_gaits = gdf['gaits_lateral'].values
            rear_gaits = gdf['gaits_rear'].values
            frames_in_clip = len(lateral_gaits)
            
            # get total #frames in each gait style for this individual for this clip
            if uniq_id in clip_total_frames.keys():
                clip_total_frames[uniq_id] = np.append(clip_total_frames[uniq_id], frames_in_clip)
            else:
                clip_total_frames[uniq_id] = frames_in_clip
            
            # get #frames where lateral legs are all down = 'stand'
            if uniq_id in clip_stand_lateral.keys():
                clip_stand_lateral[uniq_id] = np.append(clip_stand_lateral[uniq_id], np.count_nonzero(lateral_gaits=='stand'))
            else:
                clip_stand_lateral[uniq_id] = np.count_nonzero(lateral_gaits=='stand')
            
            # get #frames where one lateral leg is up = 'pentapod'
            if uniq_id in clip_pentapod.keys():
                clip_pentapod[uniq_id] = np.append(clip_pentapod[uniq_id], np.count_nonzero(lateral_gaits=='pentapod'))
            else:
                clip_pentapod[uniq_id] = np.count_nonzero(lateral_gaits=='pentapod')
                
            # get #frames where two lateral legs are up in adjacent segments on opposite sides = 'tetrapod_canonical'
            if uniq_id in clip_tetrapod_canonical.keys():
                clip_tetrapod_canonical[uniq_id] = np.append(clip_tetrapod_canonical[uniq_id], np.count_nonzero(lateral_gaits=='tetrapod_canonical'))
            else:
                clip_tetrapod_canonical[uniq_id] = np.count_nonzero(lateral_gaits=='tetrapod_canonical')
                 
            # get #frames where two lateral legs are up in same segment on opposite sides = 'tetrapod_gallop'
            if uniq_id in clip_tetrapod_gallop.keys():
                clip_tetrapod_gallop[uniq_id] = np.append(clip_tetrapod_gallop[uniq_id], np.count_nonzero(lateral_gaits=='tetrapod_gallop'))
            else:
                clip_tetrapod_gallop[uniq_id] = np.count_nonzero(lateral_gaits=='tetrapod_gallop')
                
            # get #frames where two lateral legs are up in a pattern not described above = 'tetrapod_other'
            if uniq_id in clip_tetrapod_other.keys():
                clip_tetrapod_other[uniq_id] = np.append(clip_tetrapod_other[uniq_id], np.count_nonzero(lateral_gaits=='tetrapod_other'))
            else:
                clip_tetrapod_other[uniq_id] = np.count_nonzero(lateral_gaits=='tetrapod_other')
                
            # get #frames where three lateral legs are up in adjacent segments on opposite sides = 'tripod_canonical'
            if uniq_id in clip_tripod_canonical.keys():
                clip_tripod_canonical[uniq_id] = np.append(clip_tripod_canonical[uniq_id], np.count_nonzero(lateral_gaits=='tripod_canonical'))
            else:
                clip_tripod_canonical[uniq_id] = np.count_nonzero(lateral_gaits=='tripod_canonical')
                
            # get #frames where three lateral legs are up in a pattern that is not 'canonical' = 'tripod_other'
            if uniq_id in clip_tripod_other.keys():
                clip_tripod_other[uniq_id] = np.append(clip_tripod_other[uniq_id], np.count_nonzero(lateral_gaits=='tripod_other'))
            else:
                clip_tripod_other[uniq_id] = np.count_nonzero(lateral_gaits=='tripod_other')
                
            # get #frames where more than three lateral legs are up = 'other'
            if uniq_id in clip_other_lateral.keys():
                clip_other_lateral[uniq_id] = np.append(clip_other_lateral[uniq_id], np.count_nonzero(lateral_gaits=='other'))
            else:
                clip_other_lateral[uniq_id] = np.count_nonzero(lateral_gaits=='other')
                
            # get #frames where both rear legs are down= 'stand'
            if uniq_id in clip_stand_rear.keys():
                clip_stand_rear[uniq_id] = np.append(clip_stand_rear[uniq_id], np.count_nonzero(rear_gaits=='stand'))
            else:
                clip_stand_rear[uniq_id] = np.count_nonzero(rear_gaits=='stand')
                
            # get #frames where both rear legs are up= 'hop'
            if uniq_id in clip_hop.keys():
                clip_hop[uniq_id] = np.append(clip_hop[uniq_id], np.count_nonzero(rear_gaits=='hop'))
            else:
                clip_hop[uniq_id] = np.count_nonzero(rear_gaits=='hop')
                
            # get #frames where one rear leg is up= 'step'
            if uniq_id in clip_step.keys():
                clip_step[uniq_id] = np.append(clip_step[uniq_id], np.count_nonzero(rear_gaits=='step'))
            else:
                clip_step[uniq_id] = np.count_nonzero(rear_gaits=='step')
        
        
    #### ===> finished collecting data. Build up dataframes to save
    
    #### ==> path_summaries dataframe ... info for each unique individual
    ids = sorted(clip_duration.keys())   
    treatments = [x.split('_')[0] for x in ids]
    individuals = [x.split('_')[1] for x in ids]
    dates = [x.split('_')[2] for x in ids]  
    scales = [np.mean(clip_scales[x]) for x in ids]
    lengths = [np.mean(clip_lengths[x]) for x in ids]
    areas = [np.mean(clip_areas[x]) for x in ids]
    durations = [np.sum(clip_duration[x]) for x in ids]
    distances = [np.sum(distance_traveled[x]) for x in ids]
    speed_mm = [distance / durations[i] for i, distance in enumerate(distances)]
    speed_bodylength = [distance / lengths[i] / durations[i] for i, distance in enumerate(distances)]
    cruising = [np.sum(clip_cruising[x]) * 100 / len(clip_cruising[x]) for x in ids]
    bearings = [np.sum(bearing_changes[x]) for x in ids]
    degrees_per_sec = [bearings[i] / duration for i, duration in enumerate(durations) ]
    stops = [np.sum(num_stops[x]) for x in ids]
    stops_per_sec = [stops[i] / duration for i, duration in enumerate(durations) ]
    turns = [np.sum(num_turns[x]) for x in ids]
    turns_per_sec = [turns[i] / duration for i, duration in enumerate(durations) ]
    
    path_summaries_dict = {'Identifier':ids,
                           'treatment':treatments,
                           'individual':individuals,
                           'date':dates,
                           'Scale (pixels in 1mm)':scales,
                           'Size (mm^2)':areas,
                           'Length (mm)':lengths,
                           'Duration analyzed (sec)':durations,
                           'Distance traveled (mm)':distances,
                           'Speed (mm/s)':speed_mm,
                           'Speed (body lengths / s)':speed_bodylength,
                           'Percentage of time cruising':cruising,
                           'Total bearing change (deg)':bearings,
                           'Bearing change (deg) / sec':degrees_per_sec,
                           'Number of stops': stops,
                           'Stops / sec':stops_per_sec,
                           'Number of turns': turns,
                           'Turns / sec':turns_per_sec}
    
    path_summaries_df = pd.DataFrame(path_summaries_dict)
    
    #### ==> step_summaries dataframe ... info for each unique individual
    
    ids = sorted(clip_stance_lateral.keys())
    treatments = [x.split('_')[0] for x in ids]
    individuals = [x.split('_')[1] for x in ids]
    dates = [x.split('_')[2] for x in ids]
    stance_duration_lateral = [np.mean(clip_stance_lateral[x]) for x in ids]
    swing_duration_lateral = [np.mean(clip_swing_lateral[x]) for x in ids]
    gait_cycle_lateral = [np.mean(clip_gait_lateral[x]) for x in ids]
    duty_factor_lateral = [np.mean(clip_duty_lateral[x]) for x in ids]
    distance_per_step_lateral = [np.mean(clip_pixels_per_step_lateral[x]) / clip_scales[x] for x in ids]
    bodylength_per_step_lateral = [np.mean(clip_pixels_per_step_lateral[x]) / np.median(clip_lengths[x]) for x in ids]
    stance_duration_rear = [np.mean(clip_stance_rear[x]) for x in ids]
    swing_duration_rear = [np.mean(clip_swing_rear[x]) for x in ids]
    gait_cycle_rear = [np.mean(clip_gait_rear[x]) for x in ids]
    duty_factor_rear = [np.mean(clip_duty_rear[x]) for x in ids]
    distance_per_step_rear = [np.mean(clip_pixels_per_step_rear[x]) / clip_scales[x] for x in ids]
    bodylength_per_step_rear = [np.mean(clip_pixels_per_step_rear[x]) / np.median(clip_lengths[x]) for x in ids]
    anterior_offsets = [np.mean(clip_anterior_offset[x]) for x in ids]
    normalized_anterior_offsets = [np.mean(clip_anterior_offset_normalized[x]) for x in ids]
    opposite_offsets_lateral = [np.mean(clip_opposite_offset_lateral[x]) for x in ids]
    opposite_offsets_lateral_normalized = [np.mean(clip_opposite_offset_lateral_normalized[x]) for x in ids]
    metachronal_lag = [np.mean(clip_metachronal_lag[x]) for x in ids]     
    metachronal_lag_normalized = [np.mean(clip_metachronal_lag_normalized[x]) for x in ids] 
    opposite_offsets_rear = [np.mean(clip_opposite_offset_rear[x]) for x in ids] 
    opposite_offsets_rear_normalized = [np.mean(clip_opposite_offset_rear_normalized[x]) for x in ids] 
        
    # WORKING HERE
    step_summaries_dict = {'Identifier':ids,
                           'treatment':treatments,
                           'individual':individuals,
                           'Stance duration (lateral legs)':stance_duration_lateral,
                           'Swing duration (lateral legs)':swing_duration_lateral,
                           'Gait cycle (lateral legs)':gait_cycle_lateral,
                           'Duty factor (lateral legs)':duty_factor_lateral,
                           'mm per step (lateral legs)':distance_per_step_lateral,
                           'bodylength per step (lateral legs)':bodylength_per_step_lateral,
                           'Stance duration (rear legs)':stance_duration_rear,
                           'Swing duration (rear legs)':swing_duration_rear,
                           'Gait cycle (rear legs)':gait_cycle_rear,
                           'Duty factor (rear legs)':duty_factor_rear,
                           'mm per step (rear legs)':distance_per_step_rear,
                           'bodylength per step (rear legs)':bodylength_per_step_rear,
                           'Metachronal lag (lateral legs)':metachronal_lag,
                           'Metachronal lag (normalized, lateral legs)':metachronal_lag_normalized,
                           'Anterior swing offsets (lateral legs)':anterior_offsets,
                           'Anterior swing offsets (normalized, lateral legs)':normalized_anterior_offsets,
                           'Opposite swing offsets (lateral legs)':opposite_offsets_lateral,
                           'Opposite swing offsets (normalized, lateral legs)':opposite_offsets_lateral_normalized,
                           'Opposite swing offsets (rear legs)':opposite_offsets_rear,
                           'Opposite swing offsets (rear, lateral legs)':opposite_offsets_rear_normalized
                           }
    
    step_summaries_df = pd.DataFrame(step_summaries_dict)
   
    #### ==> gait_summaries dataframe ... info for each unique individual
    ids = sorted(clip_total_frames.keys())
    treatments = [x.split('_')[0] for x in ids]
    individuals = [x.split('_')[1] for x in ids]
    dates = [x.split('_')[2] for x in ids]
    num_frames = [np.sum(clip_total_frames[x]) for x in ids]
    frames_standing_lateral = [np.sum(clip_stand_lateral[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_pentapod = [np.sum(clip_pentapod[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_tetrapod_canonical = [np.sum(clip_tetrapod_canonical[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_tetrapod_gallop = [np.sum(clip_tetrapod_gallop[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_tetrapod_other = [np.sum(clip_tetrapod_other[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_tripod_canonical = [np.sum(clip_tripod_canonical[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_tripod_other = [np.sum(clip_tripod_other[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_other = [np.sum(clip_other_lateral[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_stand_rear = [np.sum(clip_stand_rear[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_hop = [np.sum(clip_hop[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    frames_step = [np.sum(clip_step[x]) * 100 / num_frames[i] for i,x in enumerate(ids)]
    
    gait_summaries_dict = {'Identifier':ids,
                           'treatment':treatments,
                           'individual':individuals,
                           'date':dates,
                           'Number of frames':num_frames,
                           '% standing (lateral legs)':frames_standing_lateral,
                           '% pentapod (lateral legs)':frames_pentapod,
                           '% tetrapod canonical (lateral legs)':frames_tetrapod_canonical,
                           '% tetrapod gallop (lateral legs)':frames_tetrapod_gallop,
                           '% tetrapod other (lateral legs)':frames_tetrapod_other,
                           '% tripod canonical (lateral legs)':frames_tripod_canonical,
                           '% tripod other (lateral legs)':frames_tripod_other,
                           '% other(lateral legs)':frames_other,
                           '% stand (rear legs)':frames_stand_rear,
                           '% hop (rear legs)':frames_hop,
                           '% step (rear legs)':frames_step
                           }
    
    gait_summaries_df = pd.DataFrame(gait_summaries_dict)
    
    # save dataframes to output file
    print('\nCombining data from all clips into ' + out_file)
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer: 
        if len(path_summaries_df) > 0:
            path_summaries_df.to_excel(writer, index=False, sheet_name='path_summaries')
        if len(step_timing_combined_df) > 0:
            step_timing_combined_df.to_excel(writer, index=False, sheet_name='step_timing')
        if len(step_summaries_df) > 0:
            step_summaries_df.to_excel(writer, index=False, sheet_name='step_summaries')
        if len(gait_summaries_df) > 0:
            gait_summaries_df.to_excel(writer, index=False, sheet_name='gait_summaries')


def addColtoDF(df, colname, st):
    st_stem = st.split('.')[0]
    num_rows = df.shape[0]
    st_column = [st_stem] * num_rows
    df[colname] = st_column
    return df    
    
def get_clips():
    
    mov_files = gaitFunctions.getFileList(['mov','mp4'])
    mov_filestems = [x.split('.')[0] for x in mov_files]
    excel_files = glob.glob('*.xlsx')
    excel_filestems = [x.split('.')[0] for x in excel_files]
    
    problem_files = []
    ok_files = []
    for mov_stem in mov_filestems:
        if mov_stem in excel_filestems:
            ok_files.append(mov_stem)
        else:
            problem_files.append(mov_stem)
    
    if len(problem_files) > 0:
        print('These movies do not have excel files:\n')
        print('\n  '.join(problem_files))
        
    return ok_files

if __name__ == '__main__':
    main()
