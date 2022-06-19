#!/usr/bin/python
from gait_analysis import *

# have output file from save_step_data.py
# read it into a list
ifile = 'all_step_data.csv'
with open(ifile, 'r') as f:
    header = f.readlines()[0].rstrip()

with open(ifile, 'r') as f:
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
all_legs = ['R4','R3','R2','R1','L1','L2','L3','L4']

# add appropriate info to header
for leg in all_legs:
    header += ',' + leg + '_mid_swings'

# set up an output file
ofile = ifile.split('.')[0] + '_swings.csv'
o2 = open(ofile,'w')

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
            converted_result = np.around( (result - step_start) / gait_cycle,4 )

            if len(converted_result) > 0:
                mid_swing_string += leg + ':' + ';'.join([str(x) for x in converted_result]) + ','
            else:
                mid_swing_string += leg + ':,'

        else: # no data for this leg
            mid_swing_string += leg + ':,'
            continue

    # finished going through legs - print out line!
    o2.write(d.rstrip() + mid_swing_string[:-1] + '\n') # remove last comma

# finished printing, close output file
o2.close()