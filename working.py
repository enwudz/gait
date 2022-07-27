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





# def get_swing_categories():
#     '''
#     function to get dictionary of different categories of swinging leg combinations
#         keys = category names
#         values = leg combinations in each category

#     Swing leg combinations:
#     Tripod gait (3 legs swinging)
#         Tripod_canonical (swing = R1L2R3 & L1R2L3)
#         Tripod_other (any other combo of 3 legs down & 3 legs up)
#     Tetrapod gait (2 legs swinging)
#         Tetrapod_canonical (swing = R1L2, R2L1, R2L3, L2R3, R3L1, L3R1)
#         Tetrapod_gallop (swing = R1L1, R2L2, R3L3)
#         Tetrapod_other (any other combo of 4 legs down & 2 legs up)
#     Pentapod (1 leg swinging) ... Nirody calls this 'wave'
#     Stand (no legs swinging)
#     Other (4 or 5 legs swinging!)
    
#     '''
#     leg_combo_keys = ['tripod_canonical', 'tripod_other', 'tetrapod_canonical', 'tetrapod_gallop', 'tetrapod_other',
#     'pentapod', 'stand', 'other'
#     ]

#     leg_combo_keys = ['contralateral pairs',
#                       'ipsilateral adjacents',
#                       'ipsilateral skips',
#                       'contralateral adjacents',
#                       'tripod',
#                       'single legs',
#                       'no legs']
#     leg_combo_values = [['L1_R1', 'L2_R2', 'L3_R3'],
#                         ['L1_L2', 'L2_L3', 'R1_R2', 'R2_R3'],
#                         ['R1_R3', 'L1_L3'],
#                         ['L1_R2', 'L2_R3', 'L3_R1', 'L2_R1', 'L3_R2', 'L1_R3'],
#                         ['L1_L3_R2', 'L2_R1_R3'],
#                         ['L1', 'L2', 'L3', 'R1', 'R2', 'R3'],
#                         ['none']]
#     swing_categories = dict(zip(leg_combo_keys, leg_combo_values))
#     return swing_categories

# def get_swing_combo_counts(leg_swing_counts, swing_categories):
#     # any other combo = 'other' and print out report of how many in which class, sorted descending
#     # question - count subsets? i.e. for L3_R1_R2, also count as L3_R1, L3_R2, R1_R2?

#     leg_swing_combos = {}

#     for k in swing_categories.keys():
#         leg_swing_combos[k] = 0

#     found_combos = []

#     # count frames for each leg combo            
#     for leg_combo in sorted(leg_swing_counts.keys()):

#         for swing_category in swing_categories.keys():
#             if leg_combo in swing_categories[swing_category]:
#                 # print(leg_combo, ' is in ', swing_categories[swing_category])
#                 leg_swing_combos[swing_category] += leg_swing_counts[leg_combo]
#                 found_combos.append(leg_combo)

#     # quantify leg combinations that are not in swing_categories
#     for leg_combo in sorted(leg_swing_counts.keys()):

#         if leg_combo not in found_combos:
#             leg_swing_combos[leg_combo] = leg_swing_counts[leg_combo]

#     return leg_swing_combos

# def get_gait_categories():
#     categories = ['stand', 'pentapod', 'tetrapod', 'tripod', 'gallop', 'other']
#     leg_groups = [['none'],
#                   ['L1', 'L2', 'L3', 'R1', 'R2', 'R3'],
#                   ['L1_R2', 'L2_R3', 'L3_R1', 'L1_R3', 'L2_R1', 'L3_R2'],
#                   ['L1_L3_R2', 'L2_R1_R3'],
#                   ['L1_R1', 'L2_R2', 'L3_R3'],
#                   ['other']
#                   ]
#     return categories, leg_groups

# # function that will take a dictionary and return proportions in gait categories
# def get_proportions_in_swing_categories(data_dictionary, num_frames):

#     tetrapod = data_dictionary['contralateral adjacents'] / num_frames
#     stand = data_dictionary['no legs'] / num_frames
#     pentapod = data_dictionary['single legs'] / num_frames
#     # wave = (data_dictionary['single legs'] + data_dictionary['no legs']) / num_frames
#     tripod = data_dictionary['tripod'] / num_frames
#     gallop = data_dictionary['contralateral pairs'] / num_frames
#     # unclassified = 1 - (tetrapod + wave + tripod + gallop)
#     other = 1 - (tetrapod + stand + pentapod + tripod + gallop)
#     # category_data = [tetrapod, wave, tripod, gallop, unclassified]
#     category_data = [stand, pentapod, tetrapod, tripod, gallop, other]

#     categories, leg_groups = get_gait_categories()

#     return categories, category_data


# function to get / combine data for the selected movie folders
# def get_swing_combo_data(movie_folders, legs):
#     swing_categories = get_swing_categories()
#     all_experiment_data = {}
#     total_frames = 0

#     for movie_folder in movie_folders:
        
#         # get frame_times for this movie (in milliseconds, e.g. [0 33 66 100 .... ])
#         frame_times = get_frame_times(movie_folder)
#         total_frames += len(frame_times)

#         # get dictionary of up & down timing for this video clip
#         # keys = leg['u'] or leg['d'] where leg is in ['L4','L3','L2','L1' (or rights)]
#         up_down_times, latest_event = getUpDownTimes(os.path.join(movie_folder, 'mov_data.txt'))

#         # get matrix of up (1's) and down (0's) data for all legs
#         # rows = legs
#         # columns = frames of video
#         leg_matrix = make_leg_matrix(legs, up_down_times, frame_times)

#         # get dictionary of #frames swinging for different combinations of legs 
#         leg_swing_counts = get_leg_swing_counts(leg_matrix, legs)

#         # get counts of #frames in each type of swing category
#         leg_swing_combos = get_swing_combo_counts(leg_swing_counts, swing_categories)

#         # add leg_swing_combos data to existing dictionary for all experiments
#         all_experiment_data = add_counts_to_dictionary(leg_swing_combos, all_experiment_data)

#     return all_experiment_data, total_frames
