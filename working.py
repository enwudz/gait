#!/usr/bin/python
from gait_analysis import *


def calculateAverages(data_folder, step_data_file, leg_list):

    # I think this is better done in a jupyter notebook, and is largely already done there in average_step_plots.ipynb
    step_averages_file = os.path.join(data_folder, 'step_averages.txt')
    o3 = open(step_averages_file, 'w')

    # read in file to pandas dataframe
    step_data = pd.read_csv(step_data_file, index_col=None)
    # print(step_data.head(3))

    # get dictionaries of anterior and opposite legs
    opposite_dict, anterior_dict = getOppAndAntLeg()

    # for each leg, get averages of step data information
    # set up a dictionary to store these vectors of leg averages
    leg_averages = {}

    header = 'leg,num_cycles,avg_stance,avg_swing,avg_cycle,avg_duty,avg_offset_anterior,avg_offset_contra'
    print(header)
    o3.write(header + '\n')

    # leg_list = ['L3'] # testing
    for leg in leg_list:
        # initialize a data vector for this leg
        leg_data_vec = np.zeros(7)

        # print(leg)
        # get all the steps for this leg
        steps_for_this_leg = step_data[step_data['ref_leg']==leg]
        # print(steps_for_this_leg)

        # number of complete gait cycles
        num_cycles = steps_for_this_leg.shape[0]
        leg_data_vec[0] = num_cycles

        # average stance duration
        avg_stance = np.around(np.mean(steps_for_this_leg['stance_time']),4)
        leg_data_vec[1] = avg_stance

        # average swing duration
        avg_swing = np.around(np.mean(steps_for_this_leg['swing_time']),4)
        leg_data_vec[2] = avg_swing

        # average gait cycle duration
        avg_cycle = np.around(np.mean(steps_for_this_leg['gait_cycle']),4)
        leg_data_vec[3] = avg_cycle

        # average duty factor
        avg_duty = np.around(np.mean(steps_for_this_leg['duty_factor']),4)
        leg_data_vec[4] = avg_duty

        # average anterior offset
         # get values for mid-swings for THIS leg
        this_leg_midswing_column = leg + '_mid_swings'
        this_leg_midswing_values = steps_for_this_leg[this_leg_midswing_column].values

        # get values for mid-swings for ANTERIOR leg
        anterior_leg = anterior_dict[leg]
        anterior_leg_midswing_column = anterior_leg + '_mid_swings'
        anterior_leg_midswing_values = steps_for_this_leg[anterior_leg_midswing_column].values

        # get values for mid-swings for CONTRALATERAL leg 
        opposite_leg = opposite_dict[leg]
        opposite_leg_midswing_column = opposite_leg + '_mid_swings'
        opposite_leg_midswing_values = steps_for_this_leg[opposite_leg_midswing_column].values

        # get values for gait_cycle_duration for THIS leg
        this_leg_gait_cycles = steps_for_this_leg['gait_cycle'].values

        anterior_offsets = []
        opposite_offsets = []

        for i, val in enumerate(this_leg_midswing_values):
            this_leg_midswing = float(val.split(':')[1])

            # get ANTERIOR midswing data for all swings during each gait cycle
            anterior_leg_midswing_data = anterior_leg_midswing_values[i].split(':')[1]

            if len(anterior_leg_midswing_data) > 0:
                if ';' in anterior_leg_midswing_data:
                    anterior_midswings = [float(x) for x in anterior_leg_midswing_data.split(';')]
                else:
                    anterior_midswings = [float(anterior_leg_midswing_data)]
                
                # calculate anterior offsets from anterior midswings
                for swing in anterior_midswings:
                    if swing >= this_leg_midswing:
                        anterior_offsets.append(swing - this_leg_midswing)
                    else: 
                        anterior_offsets.append(swing + this_leg_gait_cycles[i] - this_leg_midswing)

            ## get CONTRALATERAL midswing data for all swings during each gait cycle
            opposite_leg_midswing_data = opposite_leg_midswing_values[i].split(':')[1]

            if len(opposite_leg_midswing_data) > 0:
                if ';' in opposite_leg_midswing_data:
                    opposite_midswings = [float(x) for x in opposite_leg_midswing_data.split(';')]
                else:
                    opposite_midswings = [float(opposite_leg_midswing_data)]
                
                # calculate anterior offsets from anterior midswings
                for swing in opposite_midswings:
                    if swing >= this_leg_midswing:
                        opposite_offsets.append(swing - this_leg_midswing)
                    else: 
                        opposite_offsets.append(swing + this_leg_gait_cycles[i] - this_leg_midswing)

        avg_anterior_offset = np.around(np.mean(anterior_offsets),4)
        leg_data_vec[5] = avg_anterior_offset

        avg_opposite_offset = np.around(np.mean(opposite_offsets),4)
        leg_data_vec[6] = avg_opposite_offset

        str_to_write = leg + ',' + ','.join([str(x) for x in leg_data_vec])
        print(str_to_write)
        o3.write(str_to_write + '\n')
    o3.close()