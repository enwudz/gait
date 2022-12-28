#!/usr/bin/python

import numpy as np
import sys
import gaitFunctions
import pandas as pd

# this script collects step information for EACH(!) COMPLETE gait cycle
#   stance length, swing length, gait cycle duration, duty factor
# this script also finds the 'mid-swing' time for each of these gait cycles 
# ... where mid-swing = halfway between leg-up and leg-down
# this script ALSO finds the timing when ALL other legs are ...
# ....in mid-swing during the gait cycle of a particular leg
# ... and expresses these times as FRACTIONS of this leg's gait cycle
# this script ALSO finds averages for all step parameters across all gait cycles for each leg

# this script ALSO gets and saves gait styles for each frame of the movie!

def main(movie_file):

    add_swing = True # do we want to collect mid-swing times for all other legs for each step?

    # load mov_data = a dictionary of UP and DOWN times for each leg
    mov_data, excel_filename = gaitFunctions.loadMovData(movie_file)

    # collect step data from mov_data
    up_down_times, last_event = gaitFunctions.getUpDownTimes(mov_data)

    '''
    For each step of each leg, we are going to collect this info: 
    legID DownTime UpTime stance swing gait duty midSwingTime
    '''
    header = 'legID,DownTime,UpTime,stance,swing,gait,duty,midSwingTime'

    # get legs
    legs = gaitFunctions.get_leg_combos()['legs_all']

    #### go through all legs, collect data for each step, and save all information in a list of lines
    data_for_steps = []
    for ref_leg in legs: 

        # ref_leg is just a single leg ... we look at one leg at a time.
        # it is called ref_leg because we are going to timing of swings (e.g. mid, initiation) 
        # for ALL other legs that swing during the gait cycle of this ref_leg
        # these times are expressed as the FRACTION of the gait cycle for each step of the ref_leg

        if ref_leg not in up_down_times.keys():
            print('No data for ' + ref_leg)
            continue

        # get timing and step characteristics for all gait cycles for this leg
        downs = up_down_times[ref_leg]['d']
        ups = up_down_times[ref_leg]['u']
        downs, ups, stance_times, swing_times, gait_cycles, duty_factors, mid_swings = gaitFunctions.getStepSummary(downs,ups)

        # go through each down step for this leg
        for i,step in enumerate(downs[:-1]): 
            
            # there needs to be one more down than up because we want the timing of COMPLETE gait cycles
            # in other words ... #downs = #ups + 1
            # and the order needs to be corect
            # e.g. down-up down-up down-up down

            # get and print information for this step
            step_stats = ','.join(str(x) for x in [ref_leg,step,ups[i],stance_times[i],swing_times[i],
                                                gait_cycles[i],duty_factors[i],mid_swings[i]])
            data_for_steps.append(step_stats)

    if add_swing is True: # do we want the mid-swing information for all other legs?
        print('Saving mid-swing times ... ')
        data_for_steps_with_swings = []

        # have output list (data_for_steps) from the code above

        # from this list, make two dictionaries of swing timing: mid_swing_dict
        # mid_swing_dict = a dictionary of leg:[mid_swings]
        # start_swing_dict = a dictionary of leg:[leg_ups aka swing_starts]
        mid_swing_dict = {}
        start_swing_dict = {}
        
        # get dictionaries of anterior and opposite legs
        opposite_dict, anterior_dict = gaitFunctions.getOppAndAntLeg()
        
        for d in data_for_steps:
            stuff = d.rstrip().split(',')
            leg = stuff[0]
            mid_swing = float(stuff[7])
            start_swing = float(stuff[2])

            if leg in mid_swing_dict.keys():
                mid_swing_dict[leg].append(mid_swing)
            else:
                mid_swing_dict[leg] = [mid_swing]

            if leg in start_swing_dict.keys():
                start_swing_dict[leg].append(start_swing)
            else:
                start_swing_dict[leg] = [start_swing]

        # for each step, add mid_swing data for all other legs
        # ------> mid_swing data is scaled as a FRACTION of the gait_cycle <------
        # add appropriate info to header
        for leg in legs:
            header += ',' + leg + '_mid_swings'

        # for each step, add start_swing data for the ANTERIOR leg and for the CONTRALATERAL leg
        # ------> start_swing data is scaled as a FRACTION of the gait_cycle <------
        
        # add appropriate info to header
        header += ',anterior_swing_start,contralateral_swing_start'

        # for each step (defining a gait cycle), get mid-swing timing of all other legs
        # where timing is defined as when the mid-swing occurs during the gait cycle of the reference leg
        # expressed as a decimal or fraction of the gait cycle of the reference leg

        # NOTE lots of duplicate code below - should probably write functions for some of this
        for d in data_for_steps:
            stuff = d.rstrip().split(',')
            ref_leg = stuff[0]
            step_start = float(stuff[1])
            gait_cycle = float(stuff[5])
            step_end = step_start + gait_cycle

            output_string = ','

            # go through ALL legs and get timing of their mid-swings
            for leg in legs:
                if leg in mid_swing_dict.keys():

                    # get mid_swing times for this leg
                    mid_swings = np.array(mid_swing_dict[leg])

                    # which of these mid_swing times are within step_start and step_end?
                    result = mid_swings[np.where(np.logical_and(mid_swings >= step_start, mid_swings <= step_end))]

                    # convert results to fraction of step_cycle
                    converted_result = np.around((result - step_start) / gait_cycle, 4)

                    if len(converted_result) > 0:
                        # if any converted result is 1 (end of gait cycle), change to 0 (beginning of gait cycle)
                        converted_result = sorted([0 if x==1 else x for x in converted_result ])
                        output_string += leg + ':' + ';'.join([str(x) for x in converted_result]) + ','
                    else:
                        output_string += leg + ':,'

                else:  # no data for this leg
                    output_string += leg + ':,'
                    continue

            # for leg that is ANTERIOR to ref_leg, get timing of swing starts (leg ups)
            anterior_leg = anterior_dict[ref_leg]
            if anterior_leg in start_swing_dict.keys():
                # get start_swing times for this leg
                start_swings = np.array(start_swing_dict[anterior_leg])
                # which of these mid_swing times are within step_start and step_end?
                result = start_swings[np.where(np.logical_and(start_swings >= step_start, start_swings <= step_end))]
                # convert results to fraction of step_cycle
                converted_result = np.around((result - step_start) / gait_cycle, 4)

                if len(converted_result) > 0:
                    # if any converted result is 1 (end of gait cycle), change to 0 (beginning of gait cycle)
                    converted_result = sorted([0 if x==1 else x for x in converted_result ])
                    output_string += anterior_leg + ':' + ';'.join([str(x) for x in converted_result]) + ','
                else:
                    output_string += anterior_leg + ':,'             
            else: # no data for this leg
                output_string += anterior_leg + ':,'

            # for leg that is CONTRALATERAL to ref_leg, get timing of swing starts (leg ups)
            opposite_leg = opposite_dict[ref_leg]
            if opposite_leg in start_swing_dict.keys():
                # get start_swing times for this leg
                start_swings = np.array(start_swing_dict[opposite_leg])
                # which of these mid_swing times are within step_start and step_end?
                result = start_swings[np.where(np.logical_and(start_swings >= step_start, start_swings <= step_end))]
                # convert results to fraction of step_cycle
                converted_result = np.around((result - step_start) / gait_cycle, 4)
                if len(converted_result) > 0:
                    # if any converted result is 1 (end of gait cycle), change to 0 (beginning of gait cycle)
                    converted_result = sorted([0 if x==1 else x for x in converted_result ])
                    output_string += opposite_leg + ':' + ';'.join([str(x) for x in converted_result]) + ','
                else:
                    output_string += opposite_leg + ':,'             
            else: # no data for this leg
                output_string += opposite_leg + ':,'

            # finished going through legs - add data to the string for this step
            step_data_string = d.rstrip() + output_string[:-1]  # [:-1] is removing last comma
            data_for_steps_with_swings.append(step_data_string)
        
        ## whew! now we have all the step data!
        #     there is a header with all of the info
        #     .... and the step data is list of lines for each step
        #     .... where each line is has values separated by commas
        
        ## can print the data out below (or modify this code to save to a file)
        # print(header)
        # for step in data_for_steps_with_swings:
        #     print(step)
            
        ## can convert the data into a pandas dataframe!
        columns = header.split(',')
        step_data_df = pd.DataFrame([line.split(',') for line in data_for_steps_with_swings], columns=columns)
        
        # if we have data for speed at each frame, determine speed for each step of each leg
        # and add this column to the dataframe
        pathtracking_df = pd.read_excel(excel_filename, sheet_name='pathtracking', index_col=None)
        if 'speed' in pathtracking_df.columns:
            step_data_df = getSpeedForStep(step_data_df, pathtracking_df)
        
        ## Save the dataframe to the excel file, in the step_timing sheet
        with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            step_data_df.to_excel(writer, index=False, sheet_name='step_timing')
            
        ## Calculate average step parameters for each leg, and write to excel file
        saveStepStats(step_data_df, excel_filename)

        # get and save gait styles for every frame
        gaitFunctions.saveGaits(movie_file)
        
        return step_data_df

    
def saveGaitStyles(up_down_times, excel_filename):
    
    # put stuff from TEMP here
    
    return
    
def saveStepStats(step_data_df, excel_filename):
    
    legs = gaitFunctions.get_leg_combos()['legs_all']
    
    stance_time = []
    swing_time = []
    gait_cycle = []
    duty_factor = []
    distances = []
    
    for leg in legs:
        stance_time.append(np.mean([float(x) for x in step_data_df[step_data_df.legID==leg]['stance'].values]))
        swing_time.append(np.mean([float(x) for x in step_data_df[step_data_df.legID==leg]['swing'].values]))
        gait_cycle.append(np.mean([float(x) for x in step_data_df[step_data_df.legID==leg]['gait'].values]))
        duty_factor.append(np.mean([float(x) for x in step_data_df[step_data_df.legID==leg]['duty'].values]))
        distances.append(np.mean([float(x) for x in step_data_df[step_data_df.legID==leg]['distance_during_step'].values]))
        
    d = {'leg':legs, 'mean stance':stance_time, 'mean swing':swing_time,
          'mean gait cycle':gait_cycle, 'mean duty factor':duty_factor,
          'mean distance':distances}
    
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='step_stats')

def getSpeedForStep(step_data_df, pathtracking_df):
    '''
    From a dataframe of step parameters for each step
    and a dataframe of path parameters for each video frame (including speed)
    return the step parameters dataframe, updated with columns for speed and distance

    Parameters
    ----------
    step_data_df : pandas dataframe
        data generated by analyzeSteps.py (this program)
        each row = a step, with stance/swing/gait/duty values, 
        ... and timing of swings of other legs with respect to this leg
    pathtracking_df : pandas dataframe
        from excel file associated with a particular video clip
        sheet = pathfinding
        data generated by analyzePath.py

    Returns
    -------
    step_data_df : pandas datarame
        input dataframe with a new columns added for speed and distance during each step

    '''
    
    # extract data from input dataframes
    frametimes = pathtracking_df.times.values
    speeds = pathtracking_df.speed.values
    distances = pathtracking_df.distance.values
    
    downs = step_data_df.DownTime.values
    gait_durations = step_data_df.gait.values
    
    # make empty vectors for step_speed and step_distance
    step_speed = np.zeros(len(downs))
    step_distance = np.zeros(len(downs))
    
    # go through each step (down)
    for i, step_start in enumerate(downs):
        
        # convert step_start to float
        step_start = float(step_start)

        # find index in time that is equal to or greater than the beginning of this step
        start_time_index = np.where(frametimes>=step_start)[0][0]
        
        # to find the time when this step ends, add gait_duration to beginning
        step_end = step_start + float(gait_durations[i])
        
        # find the index in time that is equal to or greater than the end of this step
        end_time_index = np.where(frametimes>=step_end)[0][0]
        
        # use these indices to get the speeds in all frames between the beginning and end of these steps
        speeds_during_step = speeds[start_time_index:end_time_index]
        
        # calculate the average speed during this step
        average_speed_during_step = np.mean(speeds_during_step)
        
        # add the value of this average speed to step_speed
        step_speed[i] = average_speed_during_step
        
        # calculate the distance traveled during this step
        distance_traveled_during_step = np.sum(distances[start_time_index:end_time_index])
        
        # add this distance to to step_distance
        step_distance[i] = distance_traveled_during_step
    
    # update step_data_df with the new columns for speed and distance
    step_data_df['speed_during_step'] = step_speed
    step_data_df['distance_during_step'] = step_distance
    
    return step_data_df

if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
       movie_file = gaitFunctions.select_movie_file()
       
    print('Movie is ' + movie_file)

    main(movie_file)