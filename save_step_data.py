#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import sys
from gait_analysis import *

# this script collects step information for EACH(!) COMPLETE gait cycle
#   stance length, swing length, gait cycle duration, duty factor
# this script also finds the 'mid-swing' time for each gait cycle 
# ... where mid-swing = halfway between leg-up and leg-down
# this script ALSO finds the timing when ALL other legs are in mid-swing during the gait cycle of a leg
# ... and expresses these times as FRACTIONS of the leg's gait cycle
# this script ALSO finds averages for all step parameters across all gait cycles for each leg

# see an older version archived_3Dec21 which did:
# for each step, print out all the step stats, and info about timing of other legs relative to that step

def main(data_folder):

    add_swing = True # do we want to collect mid-swing times for all other legs for each step?

    # get data ... which folder should we look in?
    # run this script in a directory that has directories containing data for clips
    if len(data_folder) == 0: 
        # get list of directories in working folder
        dirs = listDirectories()
        # select a SINGLE directory that contains data to analyze
        data_folder = selectOneFromList(dirs)
    mov_data = os.path.join(data_folder, 'mov_data.txt')
    fileTest(mov_data)

    # collect step data from mov_data.txt
    up_down_times, movieLength = getUpDownTimes(mov_data)

    '''
    stepDataLines: 
    legID DownTime UpTime stance stride gait duty midSwingTime
    '''

    # initialize an empty leg_dict dictionary to store data for each leg
    # leg_dict['leg']['datatype'] = list of values
    leg_dict = makeLegDict()

    # make an output file to save data for each step!
    ofile = os.path.join(data_folder, 'all_step_data.csv')
    o = open(ofile,'w')

    # write a header row to the output file
    header = 'ref_leg,down_time,up_time,stance_time,swing_time,gait_cycle,duty_factor,mid_swing'
    o.write(header + '\n')

    # define categories of legs
    lateral_legs = ['R3','R2','R1','L1','L2','L3']
    rear_legs = ['R4','L4']
    all_legs = [rear_legs[0]] + lateral_legs + [rear_legs[1]]

    # go through all legs, collect data for each step, and write to output file
    for ref_leg in all_legs: 

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
        downs, ups, stance_times, swing_times, gait_cycles, duty_factors, mid_swings = getStepSummary(downs,ups)

        # go through each down step for this leg
        for i,step in enumerate(downs[:-1]): 
            # there is one more down than any other value because we want the timing of COMPLETE gait cycles
            # e.g. down-up down-up down-up down

            # get and print information for this step
            step_stats = ','.join(str(x) for x in [ref_leg,step,ups[i],stance_times[i],swing_times[i],
                                                gait_cycles[i],duty_factors[i],mid_swings[i]])
            o.write(step_stats + '\n')

    o.close()

    if add_swing is True:
        print('Saving mid-swing times ... ')

        # have output file (all_step_data) from the code above
        # get the header from this file
        with open(ofile, 'r') as f:
            header = f.readlines()[0].rstrip()

        # read the rest of this file into a list called data_for_steps
        with open(ofile, 'r') as f:
            data_for_steps = [x.rstrip() for x in f.readlines()[1:]]

        # from this list, make two dictionaries of swing timing: mid_swing_dict
        # mid_swing_dict = a dictionary of leg:[mid_swings]
        # start_swing_dict = a dicationary of leg:[leg_ups aka swing_starts]
        mid_swing_dict = {}
        start_swing_dict = {}
        
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
        for leg in all_legs:
            header += ',' + leg + '_mid_swings'

        # for each step, add start_swing data for the ANTERIOR leg and for the CONTRALATERAL leg
        # ------> start_swing data is scaled as a FRACTION of the gait_cycle <------
        # get dictionaries of anterior and opposite legs
        opposite_dict, anterior_dict = getOppAndAntLeg()
        # add appropriate info to header
        header += ',anterior_swing_start,contralateral_swing_start'

        # set up an output file and write the header
        o2file = ofile.split('.')[0] + '_swings.csv'
        o2 = open(o2file, 'w')
        o2.write(header + '\n')

        # for each step (defining a gait cycle), get mid-swing timing of all other legs
        # where timing is defined as when the mid-swing occurs during the gait cycle of the reference leg
        # expressed as a decimal or fraction of the gait cycle of the reference leg

        # NOTE lots of duplicate code here - should probably write functions for some of this
        for d in data_for_steps:
            stuff = d.rstrip().split(',')
            ref_leg = stuff[0]
            step_start = float(stuff[1])
            gait_cycle = float(stuff[5])
            step_end = step_start + gait_cycle

            output_string = ','

            # go through ALL legs and get timing of their mid-swings
            for leg in all_legs:
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

            # finished going through legs - print out line!
            o2.write(d.rstrip() + output_string[:-1] + '\n')  # [:-1] is removing last comma

        # finished printing, close output file
        o2.close()


if __name__== "__main__":
    if len(sys.argv) > 1:
            data_folder = sys.argv[1]
            print('looking in ' + data_folder)
    else:
        data_folder = ''

    main(data_folder)