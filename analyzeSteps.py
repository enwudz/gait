#!/usr/bin/python

import numpy as np
import sys
import gaitFunctions
import pandas as pd
from openpyxl import load_workbook

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

    # load excel file for this movie file        
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file.split('.')[0]) 
    
    # find steptracking sheets in this excel file
    xl = pd.ExcelFile(excel_filename)
    sheets = xl.sheet_names
    steptracking_sheets = sorted([x for x in sheets if 'steptracking' in x ])
    
    if len(steptracking_sheets) == 0:
        print('No step tracking data for ' + movie_file)
        return
    else:
        print('\nAnalyzing step data for ' + movie_file)

    add_mid_swing = True # do we want to collect mid-swing times for all other legs for each step?

    # get legs
    num_feet = gaitFunctions.get_num_feet(movie_file)
    legs = gaitFunctions.get_leg_list(num_feet)

    # get pathtracking_df for this movie
    pathtracking_df = pd.read_excel(excel_filename, sheet_name='pathtracking', index_col=None)
    
    # get pathstats_df for this movie
    pathstats_df = pd.read_excel(excel_filename, sheet_name='path_stats', index_col=None)
    
    # get rid of old sheets if present step_timing, gait_styles, step_stats
    wb = load_workbook(excel_filename)
    old_sheets = ['step_timing', 'gait_styles', 'step_stats']
    for sheet in old_sheets:
        if sheet in sheets:
            print('... Removing old ' + sheet)
            wb.remove(wb[sheet])
    wb.save(excel_filename)
        
    # get steptracking bouts from steptracking sheets
    steptracking_bouts = getStepTrackingBouts(excel_filename, steptracking_sheets)
    
    '''
    Make Header and container for step data
    For each step of each leg, we are going to collect this info: 
    legID DownTime UpTime stance swing gait duty midSwingTime
    '''
    data_for_steps = []
    header = 'legID,DownTime,UpTime,stance,swing,gait,duty,midSwingTime'
    
    if add_mid_swing is True: # if we are getting mid-swing data, we need more columns in header
        print('Saving mid-swing times to step_timing sheet ... ')
        # add appropriate info to header
        data_for_steps_with_swings = []
        for leg in legs:
            header += ',' + leg + '_mid_swings'
        header += ',anterior_swing_start,contralateral_swing_start'

    for steptracking_sheet in steptracking_sheets:
        
        print('... getting steps from ' + steptracking_sheet)
        # load mov_data = a dictionary of UP and DOWN times for each leg ... or complain that this data is not available
        mov_data, excel_filename = gaitFunctions.loadUpDownData(excel_filename, steptracking_sheet)
    
        # collect step data from mov_data
        up_down_times, last_event = gaitFunctions.getUpDownTimes(mov_data)
    
        #### go through all legs, collect data for each step, and save all information in a list of lines
        for ref_leg in legs: 
    
            # ref_leg is just a single leg ... we look at one leg at a time.
            # it is called ref_leg because we are going to timing of swings (e.g. mid, initiation) 
            # for ALL other legs that swing during the gait cycle of this ref_leg
            # these times are expressed as the FRACTION of the gait cycle for each step of the ref_leg
    
            if ref_leg not in up_down_times.keys():
                badLeg(ref_leg, movie_file)
                return
    
            # get timing and step characteristics for all gait cycles for this leg
            downs = up_down_times[ref_leg]['d']
            ups = up_down_times[ref_leg]['u']
            
            downs, ups, stance_times, swing_times, gait_cycles, duty_factors, mid_swings = gaitFunctions.getStepSummary(downs,ups)
            if len(stance_times) == 0:
                badLeg(ref_leg, movie_file)
                return
            
            # if leg down the whole time, then downs = [0] and ups = [length_of_clip] 
            if len(downs) == 1 and len(ups) == 1:
                i=0
                data_for_steps.append(','.join(str(x) for x in [ref_leg,downs[0],ups[i],stance_times[i],swing_times[i],
                                                       gait_cycles[i],duty_factors[i],mid_swings[i]]))
            
            # if many steps, want to go through each down step (aside from the last one)
            else: 
    
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

    ### FINISHED COLLECTING DATA FOR STEPS ... now, do we want to also collect mid-swing times?
    if add_mid_swing is True: # do we want the mid-swing information for all other legs?

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

        # for each step, add start_swing data for the ANTERIOR leg and for the CONTRALATERAL leg
        # ------> start_swing data is scaled as a FRACTION of the gait_cycle <------

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

            # for the leg that is ANTERIOR to ref_leg, get timing of swing starts (leg ups)
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

            # for the leg that is CONTRALATERAL to ref_leg, get timing of swing starts (leg ups)
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
        
        ## Print the data out below (or modify this code to save to a file)
        # print(header)
        # for step in data_for_steps_with_swings:
        #     print(step)
        
        # Convert the data into a pandas dataframe!
        columns = header.split(',')        
        step_data_df = pd.DataFrame([line.split(',') for line in data_for_steps_with_swings], columns=columns)
        
        # if we have data for speed at each frame, determine speed for each step of each leg
        # and add this column to the dataframe
        if 'speed' in pathtracking_df.columns:
            step_data_df = getSpeedForStep(step_data_df, pathtracking_df, pathstats_df)
        
        
        # swing offsets and metachronal lag for lateral legs
        anterior_offsets, contralateral_offsets, metachronal_lag = getOffsets(step_data_df, steptracking_bouts)
        # add these to the step_data dataframe
        step_data_df['anterior_offsets'] = anterior_offsets
        step_data_df['contralateral_offsets'] = contralateral_offsets
        step_data_df['metachronal_lag'] = metachronal_lag
        
        # From step_data_df,
        # METACHRONAL LAG RATIO for each L3-R3 pair
        # and METACHRONAL BEARING CHANGE for each L3-R3 pair
        step_data_df = metachronalRatioAndBearing(step_data_df, pathtracking_df)
        
        ## Save the dataframe to the excel file, in the step_timing sheet
        with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            step_data_df.to_excel(writer, index=False, sheet_name='step_timing')
        
        ## Calculate average step parameters for each leg, and write to the step_stats sheet of the excel file
        saveStepStats(legs, step_data_df, excel_filename)

        # get gait styles for every frame and make a dataframe
        gait_df = gaitFunctions.getGaitDataframe(movie_file)
        
        # here is where we can get coordination strength
        # stances, swings = combineSteptracking(excel_filename)
        
        for gait_style in ['tetrapod','tripod']:
            coord_strength_vec, coord_speed_vec = getCssAndSpeeds(gait_df, gait_style)
            coord_col = gait_style + '_coordination'
            speed_col = gait_style + '_speed'
            gait_df[coord_col] = coord_strength_vec
            gait_df[speed_col] = coord_speed_vec
        
        print('Saving gaits to gait_styles sheet ... ')
        with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
            gait_df.to_excel(writer, index=False, sheet_name='gait_styles')
            
        # # clean up!
        # gaitFunctions.removeFramesFolder(movie_file)
        # gaitFunctions.cleanUpTrash(movie_file)
        
        # return step_data_df

def metachronalRatioAndBearing(step_data_df, pathtracking_df):
    # From step_data_df,
    # METACHRONAL LAG RATIO for each L3-R3 pair
    # and METACHRONAL BEARING CHANGE for each L3-R3 pair
    
    ## doing this a LOOPY way instead of a Pythonic vector way ... b/c I'm tired
    # so, get vectors of the data I need to do this . . . 
    legID = step_data_df.legID.values
    DownTime = [float(x) for x in step_data_df.DownTime.values]
    UpTime = [float(x) for x in step_data_df.UpTime.values]
    gait = [float(x) for x in step_data_df.gait.values]
    mcl = [float(x) for x in step_data_df.metachronal_lag.values]
    
    frametimes = np.array([float(x) for x in pathtracking_df.times.values])
    bearings = np.array([float(x) for x in pathtracking_df.filtered_bearings.values])
    
    ## make empty containers for the new data I will add
    mcl_ratio = np.full(len(legID),np.nan) # nan array for metachronal lag ratio
    mc_bearing = np.full(len(legID),np.nan) # nan array for metachronal bearing change
    
    L3_idx = np.where(legID == 'L3')[0]
    R3_idx = np.where(legID == 'R3')[0]
    
    # ## get the data!
    # # go through the L3 indices
    for idx in L3_idx:
        # does this L3 have a metachronal lag? 
        nextR = 10000
        nextR_idx = np.nan
        if mcl[idx] > 0:
            # print(legID[idx],DownTime[idx]) # testing leg identity (L3) and stance onset timing - OK
            # if yes, get next DownTime for next R3 step!
            for Ri in R3_idx:
                if DownTime[Ri] > DownTime[idx] and DownTime[Ri] < nextR:
                    nextR = DownTime[Ri]
                    nextR_idx = Ri
            # does this R3 have a metachronal lag?
            if nextR_idx > 0 and mcl[nextR_idx] > 0:
                # if yes, calculate ratio and bearing change from L3 DownTime to R3 UpTime
                # print(mcl[idx],  mcl[nextR_idx], np.log2(mcl[idx] / mcl[nextR_idx]) ) # testing OK
                mcl_ratio[idx] = np.log2(mcl[idx] / mcl[nextR_idx] )
                
                # get bearing at DownTime of L3 and at end of L3 gait cycle
                L3_DownTime = DownTime[idx]
                end_L3_cycle = DownTime[idx] + gait[idx]
                bearing_start = bearings[np.where(frametimes>=L3_DownTime)[0][0]]
                bearing_end = bearings[np.where(frametimes>=end_L3_cycle)[0][0]]
                # print(L3_DownTime,end_L3_cycle,bearing_start,bearing_end) # testing OK
                bearing_change = gaitFunctions.change_in_bearing(bearing_start,bearing_end)
                mc_bearing[idx] = bearing_change         
    
    ## add the new columns to the dataframe
    step_data_df['mcl_LR_ratio'] = mcl_ratio
    step_data_df['L3_bearing_change'] = mc_bearing    
    
    return step_data_df

def badLeg(leg, pathstats_df):
    print('\n **** No data for ' + leg + ' ****')
    print(' **** steptracking is problematic in ' + movie_file + ' ****\n')

def getStepTrackingBouts(excel_file, steptracking_sheets):

    steptracking_bouts = []
    for sheet in steptracking_sheets:
        earliest_time = 100000
        latest_time = 0
        mov_data, excel_filename = gaitFunctions.loadUpDownData(excel_file, sheet)
        for leg in mov_data.keys():
            vals = np.array([float(x) for x in mov_data[leg].split()])
            if np.min(vals) < earliest_time:
                earliest_time = np.min(vals)
            if np.max(vals) > latest_time:
                latest_time = np.max(vals)
        timerange = str(earliest_time)+'-'+str(latest_time)
        steptracking_bouts.append(timerange)
    
    return steptracking_bouts

def findBoutEnd(event_time,bouts):
    
    '''
    Parameters
    ----------
    event_time : floating point decimal
        timing of a particular event
    bouts : list
        list of different time ranges

    Returns
    -------
    bout_end : floating point decimal
        the timing of the end of the bout during which the event_time occurred

    '''
    
    for bout in bouts:
        bout_boundaries = [float(x) for x in bout.split('-')]
        bout_start = bout_boundaries[0]
        bout_end = bout_boundaries[1]
        if event_time >= bout_start and event_time <= bout_end:
            return bout_end
    
    print('could not find end of bout for event at ' + str(event_time))
    return 0

def getOffsets(step_df, steptracking_bouts): 
    '''
    Get offsets (3 to 2 swing start, 2 to 1 swing start) 
    Get metachronal lag (3 --> 1 swing starts with requirement that 2 swings in middle)
    So ... this is geared only for six legs
    
    Offsets are only recorded for legs during steptracking bouts 
        (usually while animal is 'cruising')
    
    Contralateral (opposite) offsets are only obtained for LEFT legs
    (time between left down and right down)

    Parameters
    ----------
    step_df : pandas dataframe
        loaded from 'step_timing' sheet of an excel file associated with a clip 
        (or group of clips from same individual)
        step_df is made from analyzeSteps.py, and requires data from frameStepper.py
        
    pathstats_df : pandas dataframe
        loaded from 'path_stats' sheet of an excel file associated with a clip 
        (or group of clips from same individual)
        pathstats_df is made from analyzeTrack.py, and requires data from autoTracker.py

    Returns
    -------
    anterior_offsets : numpy array
        times (in seconds) between swing starts of anterior legs for leg pairs 2 and 3
        NaN for leg pairs 1 and 4
        
    contralateral_offsets : numpy array
        times (in seconds) between swing starts of anterior legs
        
    metachronal_lag : numpy array
        times (in seconds) between swing start of 3rd leg and 1st leg (with 2nd leg swinging in middle)
        NaN for all legs other than 3rd pair
    
    '''
    
    # make vectors of steps and swing starts
    steps = step_df['legID'].values
    swing_start_array = step_df['UpTime'].values
    
    # Make swing_starts = a dictionary of swing starts (in numpy array), keyed by leg
    swing_starts = {}
    for i, step in enumerate(steps):
        if step not in swing_starts.keys():
            swing_starts[step] = np.array([float(swing_start_array[i])])
        else:
            swing_starts[step] = np.append(swing_starts[step], float(swing_start_array[i]))
    # print(swing_starts) # testing
    
    # get dictionaries of anterior and opposite legs
    opposite_dict, anterior_dict = gaitFunctions.getOppAndAntLeg()
    
    # make arrays of NaN
    anterior_offsets = np.zeros(len(steps))
    anterior_offsets[:] = np.nan
    
    contralateral_offsets = np.zeros(len(steps))
    contralateral_offsets[:] = np.nan
    
    metachronal_lag = np.zeros(len(steps))
    metachronal_lag[:] = np.nan
    
    # populate the arrays of swing offsets
    for i, leg in enumerate(steps):
        
        swing_time = float(swing_start_array[i])
        
        ## when finding the offset between legs, want to make sure that the 
        ## next step is within the SAME cruising bout
        # which cruising bout is this step in
        # when does this bout end? 
        bout_end = findBoutEnd(swing_time, steptracking_bouts)
        
        ## OPPOSITE OFFSETS
        # for all LEFT legs, enter time of next swing of opposite leg (if available)
        if 'L' in leg:
            opposite_leg = opposite_dict[leg]
            opposite_swings = swing_starts[opposite_leg]
            next_opposite_swing = get_next_event(swing_time, opposite_swings)
            if next_opposite_swing > 0 and next_opposite_swing <= bout_end:
                contralateral_offsets[i] = next_opposite_swing - swing_time
        
        ## ANTERIOR OFFSETS
        # for 2nd pair and 3rd pair, enter time of next swing of adjacent anterior leg 
        if '2' in leg or '3' in leg:
            anterior_leg = anterior_dict[leg]
            anterior_swings = swing_starts[anterior_leg]
            next_anterior_swing = get_next_event(swing_time, anterior_swings)
            if next_anterior_swing > 0 and next_anterior_swing <= bout_end:
                anterior_offsets[i] = next_anterior_swing - swing_time
                
        # METACHRONAL LAG
        # for 3rd pair, get next time of 2nd leg on same side ... 
        # THEN get next time of 1st leg on same side
        if '3' in leg:
            second_leg = anterior_dict[leg]
            second_leg_swings = swing_starts[second_leg]
            next_second_leg_swing = get_next_event(swing_time, second_leg_swings)
            # print(leg,swing_time,second_leg,next_second_leg_swing) # testing OK
            if next_second_leg_swing > 0 and next_second_leg_swing <= bout_end:
                first_leg = anterior_dict[second_leg]
                first_leg_swings = swing_starts[first_leg]
                next_first_leg_swing = get_next_event(next_second_leg_swing, first_leg_swings)
                # print(leg,swing_time,second_leg,next_second_leg_swing,first_leg,next_first_leg_swing) # testing OK
                if next_first_leg_swing > 0 and next_first_leg_swing <= bout_end:
                    # print(leg, swing_time, next_second_leg_swing, next_first_leg_swing) # testing
                    lag = next_first_leg_swing - swing_time
                    metachronal_lag[i] = lag
    
    return anterior_offsets, contralateral_offsets, metachronal_lag
    
def get_next_event(event_time, event_times):
    
    next_event_time = 0
    next_event_ind = np.where(event_times >= event_time)[0]
    if len(next_event_ind) > 0:
        next_event_time = event_times[next_event_ind[0]]
    return next_event_time
    
def saveStepStats(legs, step_data_df, excel_filename):
    
    # we only want to save the stats for steps that occur during cruise bouts
    cruise_steps = step_data_df[step_data_df.cruising_during_step==True]
    
    stance_time = []
    swing_time = []
    gait_cycle = []
    duty_factor = []
    distances = []
    
    for leg in legs:
        stance_time.append(np.mean([float(x) for x in cruise_steps[cruise_steps.legID==leg]['stance'].values]))
        swing_time.append(np.mean([float(x) for x in cruise_steps[cruise_steps.legID==leg]['swing'].values]))
        gait_cycle.append(np.mean([float(x) for x in cruise_steps[cruise_steps.legID==leg]['gait'].values]))
        duty_factor.append(np.mean([float(x) for x in cruise_steps[cruise_steps.legID==leg]['duty'].values]))
        try: # this will only work if analyzePath has already been run . . . 
            distances.append(np.mean([float(x) for x in cruise_steps[cruise_steps.legID==leg]['distance_during_step'].values]))
        except:
            distances.append(0)
        
    d = {'leg':legs, 'mean stance':stance_time, 'mean swing':swing_time,
          'mean gait cycle':gait_cycle, 'mean duty factor':duty_factor,
          'mean distance':distances}
    
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, engine='openpyxl', if_sheet_exists='replace', mode='a') as writer: 
        df.to_excel(writer, index=False, sheet_name='step_stats')

def getSpeedForStep(step_data_df, pathtracking_df, pathstats_df):
    
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
    pathstats_df : pandas dataframe
        loaded from 'path_stats' sheet of an excel file associated with a clip 
        (or group of clips from same individual)
        pathstats_df is made from analyzeTrack.py, and requires data from autoTracker.py

    Returns
    -------
    step_data_df : pandas datarame
        input dataframe with a new columns added for speed and distance during each step

    '''
    
    # extract data from pathtracking dataframe
    frametimes = pathtracking_df.times.values
    speeds = pathtracking_df.speed.values
    distances = pathtracking_df.distance.values
    stops = pathtracking_df.stops.values
    turns = pathtracking_df.turns.values
    areas = pathtracking_df.areas.values
    lengths = pathtracking_df.lengths.values
    
    # extract data from steptracking dataframe
    downs = step_data_df.DownTime.values
    gait_durations = step_data_df.gait.values
    stance_durations = step_data_df.stance.values
    
    # make empty vectors for step_speed and step_distance
    step_speed = np.zeros(len(downs))
    step_distance = np.zeros(len(downs))
    
    # make empty vectors for areas and lengths
    step_tardiareas = np.zeros(len(downs))
    step_tardilengths = np.zeros(len(downs))
    
    # make empty vector for cruising (i.e. not turning, not stopping)
    step_cruising = np.empty(len(downs), dtype=bool)
    
    # go through each step (down)
    for i, step_start in enumerate(downs):
        
        # convert step_start to float
        step_start = float(step_start)

        # find index in time that is equal to or greater than the beginning of this step
        start_time_index = np.where(frametimes>=step_start)[0][0]
        
        # to find the time when this step ends, two choices:
        # get full step cycle by adding gait_duration to step_start
        # OR only care about when this leg is 'working', and stance_duration to step_start??
        step_end = step_start + float(stance_durations[i])
        
        # find the index in time that is equal to or greater than the end of this step
        try:
            end_time_index = np.where(frametimes>=round(step_end,3))[0][0]
        except:
            end_time_index = len(frametimes)
        
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
        
        # determine whether there was a stop or a turn during this step
        stops_during_step = np.sum(stops[start_time_index:end_time_index])
        turns_during_step = np.sum(turns[start_time_index:end_time_index])
        if stops_during_step > 0 or turns_during_step > 0:
            step_cruising[i] = False
        else:
            step_cruising[i] = True
            
        # calculate the average area measured for this step
        step_tardiareas[i] = np.mean(areas[start_time_index:end_time_index])
        
        # calculate the average length measured for this step
        step_tardilengths[i] = np.mean(lengths[start_time_index:end_time_index])
    
    # get scale from pathstats_df
    scale = pathstats_df[pathstats_df['path parameter']=='scale'].values[0][1]
    # unit = pathstats_df[pathstats_df['path parameter']=='unit'].values[0][1]

    # update step_data_df with the new columns for speed and distance and cruising
    step_data_df['speed_during_step'] = step_speed
    step_data_df['speed_during_step_scaled'] = step_speed / scale
    step_data_df['distance_during_step'] = step_distance
    step_data_df['distance_during_step_scaled'] = step_distance / scale
    step_data_df['cruising_during_step'] = step_cruising
    
    # update step_data_df with new columns for areas and lengths
    step_data_df['average_tardigrade_area'] = step_tardiareas
    step_data_df['average_tardigrade_length'] = step_tardilengths
 
    return step_data_df

def combineSteptracking(excel_file):
    
    '''
    combine steptracking data from all sheets within an excel file
    into two dictionaries of stances and swings
    key = legs
    '''
    
    xl = pd.ExcelFile(excel_file)
    sheets = xl.sheet_names
    steptracking_sheets = sorted([x for x in sheets if 'steptracking' in x ])
    
    stances = {}
    swings = {}
    
    for sheet in steptracking_sheets:
        mov_data_df = xl.parse(sheet)
        leg_states = mov_data_df.leg_state.values
        leg_times = mov_data_df.times.values
        for i,state in enumerate(leg_states):
            leg = state.split('_')[0]
            if 'down' in state:
                stances_leg = np.array([float(x) for x in leg_times[i].split()])
                if leg in stances:
                    stances[leg] = np.concatenate((stances[leg], stances_leg))
                else:
                    stances[leg] = stances_leg
            elif 'up' in state:
                swings_leg = np.array([float(x) for x in leg_times[i].split()])
                if leg in swings:
                    swings[leg] = np.concatenate((swings[leg], swings_leg))
                else:
                    swings[leg] = swings_leg
                    
    # done collecting stances and swings ... put them in order
    for leg in stances.keys():
        stances[leg] = np.sort(stances[leg])
        swings[leg] = np.sort(swings[leg])
        
    return stances, swings

def truncToThree(flt):
    '''
    truncate a floating point decimal to three decimal places (not rounded)
    '''
    return int(flt*1000)/1000


def legsInCombo(legs, combo):
    '''
    Function to see if any leg in a list of legs is in a particular combo
        inputs = 
            legs = a list of legs (e.g. ['L1','R3'])
            combo = a string (e.g. 'L1_R2')
    '''
    
    try: 
        np.isnan(combo)
        return False
    
    except:
        for leg in legs:
            if leg in combo:
                return True
        return False

def css(combo_vec, start_bout, end_bout):
    
    '''
    Function to calculate coordination score based on boundaries
        input = boundaries and vector of leg combos for each frame
        Get earliest start for combo at beginning
        Get latest end for combo at end
        Calculate coordination strength
    '''
    
    # move back from start_bout to get earliest swing of legs in swinging combo
    back_counter = 0
    
    legs_at_start = combo_vec[start_bout].split('_')
    found_start = False
    
    while found_start is False:
        
        idx = start_bout - back_counter - 1
        if idx >= 0:
            
            combo = combo_vec[idx]
            if legsInCombo(legs_at_start, combo):
                back_counter += 1
            else:
                found_start = True
                
        else:
            found_start = True
    
    # move forward from end_bout to get latest swing of legs in swinging combo
    forward_counter = 0
    
    legs_at_end = combo_vec[end_bout].split('_')
    found_end = False
    
    while found_end is False:
        
        idx = end_bout + forward_counter + 1
        if idx < len(combo_vec):
            
            combo = combo_vec[idx]
            if legsInCombo(legs_at_end, combo):
                forward_counter += 1
            else:
                found_end = True
            
        else:
            found_end = True
            
    bout_duration = end_bout + 1 - start_bout
    tot_duration = bout_duration + back_counter + forward_counter
    
    coordination_strength = bout_duration / tot_duration
    
    # omit data that overlap with beginning or ending of clip
    if start_bout - back_counter > 0 and end_bout + forward_counter < len(combo_vec) - 1:
        return coordination_strength, bout_duration
    else:
        return np.nan, 0


def speedInBout(speed_vec, start_bout, end_bout):
    '''
    Function to calculate speed based on boundaries
        input = boundaries and vector of speeds for each frame
    '''
    return np.mean(speed_vec[start_bout:end_bout+1])


def common_member(a, b):
    '''
    Function to see if there is any intersection between two lists
    '''
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return(True) 
    return(False) 

def getCssAndSpeeds(gait_styles_df, gait_style): # NEW VERSION JULY 2024

    '''
    New version July 2024: 
        CSS is now robust to changes in identities of swinging legs
        For example: 
        L1R2 ==> L3R2 seamlessly is now considered same bout of tetrapod instead of two bouts

    
    Calculate 'coordination strength' for a specified gait_style ('tetrapod' or 'tripod')
    This only considers bouts of 'canonical' gait
    
    See Wosnitza et al. 2013: https://doi.org/10.1242/jeb.078139
    
    Inputs:
    gait_styles_df: DataFrame - loaded from gait_styles sheet
    gait_style: String - 'tetrapod' or 'tripod'
    
    Returns:
    a copy of gait_styles_df with two new columns:
        Coordination strength for each bout of the specified gait_style
        Average speed (in bodylengths / sec) for each bout of the specified gait_style
    '''
    
    # define searchterm
    searchterm = gait_style + '_canonical'
    
    # get vectors we need
    speed_vec = gait_styles_df['speed (bodylength/s)'].values
    gaits_vec = gait_styles_df['gaits_lateral'].values
    combo_vec = gait_styles_df['swinging_lateral'].values
    
    # set up vectors for return
    coord_strength_vec = np.empty(len(gaits_vec))
    coord_strength_vec[:] = np.nan
    coord_speed_vec = np.empty(len(gaits_vec))
    coord_speed_vec[:] = np.nan

    current_combo = ''
    begin_bout = 0
    
    for i, frame in enumerate(gaits_vec):
        if frame == searchterm: # if X_canonical

            if len(current_combo) > 0: # if have a current combo

                if frame == current_combo: # this is the same combo
                    end_bout = i # update end to this frame

                else: # this is a different combo
                    currentcombo_list = current_combo.split('_')
                    newcombo_list = combo_vec[i].split('_')

                    if common_member(currentcombo_list, newcombo_list): # If overlap with current combo
                        current_combo = combo_vec[i] # Update current combo
                        end_bout = i # update end to this frame

                    else: # No overlap with current combo, all done with old one

                        # Calculate things based on beginning and end
                        coordination_strength, bout_duration = css(combo_vec, begin_bout, end_bout)
                        if bout_duration > 0:
                            avg_bout_speed = speedInBout(speed_vec, begin_bout, end_bout)
                        else:
                            avg_bout_speed = np.nan

                        # Insert into CSS and Boutspeed vectors at beginning index
                        coord_strength_vec[begin_bout] = coordination_strength
                        coord_speed_vec[begin_bout] = avg_bout_speed

                        # testing
                        # print(begin_bout, end_bout, coordination_strength, avg_bout_speed)

                        # done ... set current combo to this new one
                        current_combo = frame

            else: # do not have a current combo, start one here
                current_combo = combo_vec[i] # Update current combo
                begin_bout = i # update beginning to this frame
                end_bout = i # update end to this frame

        else: # not X_canonical

            if len(current_combo) > 0: # if have a current combo, all done with it

                # Calculate things based on beginning and end
                coordination_strength, bout_duration = css(combo_vec, begin_bout, end_bout)
                if bout_duration > 0:
                    avg_bout_speed = speedInBout(speed_vec, begin_bout, end_bout)
                else:
                    avg_bout_speed = np.nan

                # Insert into CSS and Boutspeed vectors at beginning index
                coord_strength_vec[begin_bout] = coordination_strength
                coord_speed_vec[begin_bout] = avg_bout_speed

                # testing
                # print(begin_bout, end_bout, coordination_strength, avg_bout_speed)

                # done ... empty out current combo
                current_combo = ''
    
    # finished going through frames, return vectors
    
    return coord_strength_vec, coord_speed_vec

def calculateCoordination(gait_styles_df, gait_style, stances, swings): # OLD VERSION! PRE July 2024
    
    '''
    Calculate 'coordination strength' for a specified gait_style ('tetrapod' or 'tripod')
    This only considers bouts of 'canonical' gait
    
    See Wosnitza et al. 2013: https://doi.org/10.1242/jeb.078139
    
    Inputs:
    gait_styles_df: DataFrame - loaded from gait_styles sheet
    gait_style: String - 'tetrapod' or 'tripod'
    stances: Dictionary - keys are leg names, values are numpy arrays of stance times
    swings: Dictionary (similar to stances)
    
    Returns:
    gait_styles_df with two new columns:
        Coordination strength for each bout of the specified gait_style
        Average speed (in bodylengths / sec) for each bout of the specified gait_style
    '''
    
    # get relevant data
    gait_search = gait_style + '_canonical'
    gaits = gait_styles_df.gaits_lateral.values
    frametimes = gait_styles_df.frametimes.values
    swinging_legs = gait_styles_df.swinging_lateral.values
    speed = gait_styles_df['speed (bodylength/s)'].values

    # initialize empty vector to add to gait_styles_df
    coord_strength_vec = np.empty(len(frametimes))
    coord_strength_vec[:] = np.nan
    coord_speed_vec = np.empty(len(frametimes))
    coord_speed_vec[:] = np.nan
    
    coord_col_name = gait_style + '_coordination'
    speed_col_name = gait_style + '_speed'

    # get timing of start and end of video
    vid_start = truncToThree(frametimes[0])
    vid_end = truncToThree(frametimes[-1])
    
    leg_combo = ''
    
    # go through frames
    for i, frametime in enumerate(frametimes):
    
        # does this frame have the style we are interested in?
        if gaits[i] == gait_search:

            # is it a new combination of legs, aka a new bout?
            leg_str = swinging_legs[i]     
            if leg_str != leg_combo:
                leg_combo = leg_str
                legs = leg_str.split('_')
                lowest_swing = 10000
                bout_start = truncToThree(frametime)
                bout_end = 10000
                highest_stance = 0
                thistime = truncToThree(frametime)
                
                # for each leg, get the times that we need
                for leg in legs:
                    swing_ok = False
                    stance_ok = False
                
                    # get the swing starts at or before start of bout
                    start_swing = np.where(swings[leg] <= thistime)[0]
                    if len(start_swing) > 0:
                        swing_ok = True
                        most_recent_swing = swings[leg][np.max(start_swing)]
                        if most_recent_swing <= lowest_swing:
                            lowest_swing = most_recent_swing
                    
                    # get the stance starts at or after the end of bout
                    next_stance = np.where(stances[leg] >= thistime)[0]
                    if len(next_stance) > 0:
                        stance_ok = True
                        next_step = stances[leg][np.min(next_stance)]
                        if next_step >= highest_stance:
                            highest_stance = next_step
                        if next_step <= bout_end:
                            bout_end = next_step
                
                if swing_ok and stance_ok:
                    # now, we have the four times we need:
                    # lowest_swing, highest_swing, lowest_stance, highest_stance
                    # does the timing overlap with the start or end of the video (inefficient!)
                    if lowest_swing > vid_start and highest_stance < vid_end:
                        
                        # calculate coordination strength for this bout
                        coordination_strength = (bout_end - bout_start) / (highest_stance-lowest_swing)
                        coord_strength_vec[i] = coordination_strength
                        
                        # calculate average speed (bodylength/sec) for this bout
                        end_frame = np.max(np.where(frametimes<=bout_end)[0])
                        avg_speed = np.mean(speed[i:end_frame+1])
                        coord_speed_vec[i] = avg_speed
                        
#                         print(lowest_swing, bout_start, bout_end, highest_stance, avg_speed, i, end_frame) # testing
   
    # finished going through frames, add columns of new data to dataframe
    gait_styles_df[coord_col_name] = coord_strength_vec
    gait_styles_df[speed_col_name] = coord_speed_vec
    
    return gait_styles_df


if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
       
    print('Movie is ' + movie_file)

    main(movie_file)
