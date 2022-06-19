#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from gait_analysis import *

# see an older version archived_3Dec21 which did:
# for each step, print out all the step stats, and info about timing of other legs relative to that step

# this one is just doing the step stats for all steps
# and adding swing_data for other legs to steps
add_swing = True

'''
stepDataLines: 
legID DownTime UpTime stance stride gait duty midSwingTime
'''

# initialize an empty leg_dict dictionary to store data for each leg
# leg_dict['leg']['datatype'] = list of values
leg_dict = makeLegDict()

# get list of directories in working folder
dirs = listDirectories()

# select a SINGLE directory that contains data to analyze
data_folder = selectOneFromList(dirs)

# parse data in mov_data.txt
# print out data for each step!

ofile = os.path.join(data_folder, 'all_step_data.csv')
o = open(ofile,'w')

lateral_legs = ['R3','R2','R1','L1','L2','L3']
rear_legs = ['R4','L4']
all_legs = [rear_legs[0]] + lateral_legs + [rear_legs[1]]

# print out a header row
header = 'ref_leg,down_time,up_time,stance_time,swing_time,gait_cycle,duty_factor,mid_swing'

o.write(header + '\n')

mov_data = os.path.join(data_folder, 'mov_data.txt')
up_down_times, movieLength = getUpDownTimes(mov_data)

for ref_leg in all_legs:

    if ref_leg not in up_down_times.keys():
        print('No data for ' + ref_leg)
        continue

    # get step stats for all gait cycles for this leg
    downs = up_down_times[ref_leg]['d']
    ups = up_down_times[ref_leg]['u']
    downs, ups, stance_times, swing_times, gait_cycles, duty_factors, mid_swings = getStepSummary(downs,ups)

    # go through each down step
    for i,step in enumerate(downs[:-1]): # there is one more down than any other value

        step_stats = ','.join(str(x) for x in [ref_leg,step,ups[i],stance_times[i],swing_times[i],
                                               gait_cycles[i],duty_factors[i],mid_swings[i]])
        o.write(step_stats + '\n')

o.close()

if add_swing is True:
    print('Saving mid-swing times ... ')
    # have output file from save_step_data.py
    # read it into a list

    with open(ofile, 'r') as f:
        header = f.readlines()[0].rstrip()

    with open(ofile, 'r') as f:
        data = [x.rstrip() for x in f.readlines()[1:]]

    # from output file of save_step_data.py, make mid_swing_dict
    # a dictionary of leg:[mid_swings]
    mid_swing_dict = {}
    for d in data:
        stuff = d.rstrip().split(',')
        leg = stuff[0]
        mid_swing = float(stuff[-1])
        if leg in mid_swing_dict.keys():
            mid_swing_dict[leg].append(mid_swing)
        else:
            mid_swing_dict[leg] = [mid_swing]

    # for each step, add mid_swing data for all other legs
    # mid_swing data is scaled as a fraction of the gait_cycle

    # make list of all legs
    all_legs = ['R4', 'R3', 'R2', 'R1', 'L1', 'L2', 'L3', 'L4']

    # add appropriate info to header
    for leg in all_legs:
        header += ',' + leg + '_mid_swings'

    # set up an output file
    o2file = ofile.split('.')[0] + '_swings.csv'
    o2 = open(o2file, 'w')
    o2.write(header + '\n')

    for d in data:
        stuff = d.rstrip().split(',')
        step_start = float(stuff[1])
        gait_cycle = float(stuff[5])
        step_end = step_start + gait_cycle

        mid_swing_string = ','

        for leg in all_legs:
            if leg in mid_swing_dict.keys():
                # get mid_swing times for this leg
                mid_swings = np.array(mid_swing_dict[leg])

                # which of these mid_swing times are within step_start and step_end
                result = mid_swings[np.where(np.logical_and(mid_swings >= step_start, mid_swings <= step_end))]

                # convert results to fraction of step_cycle
                converted_result = np.around((result - step_start) / gait_cycle, 4)

                if len(converted_result) > 0:
                    mid_swing_string += leg + ':' + ';'.join([str(x) for x in converted_result]) + ','
                else:
                    mid_swing_string += leg + ':,'

            else:  # no data for this leg
                mid_swing_string += leg + ':,'
                continue

        # finished going through legs - print out line!
        o2.write(d.rstrip() + mid_swing_string[:-1] + '\n')  # remove last comma

    # finished printing, close output file
    o2.close()