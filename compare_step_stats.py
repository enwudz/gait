#!/usr/bin/python
from gait_analysis import *
from plotTools import *
import os

# run from within analyzed_movies folder

def fileToArray(path_to_file):
    with open(path_to_file) as f:
        data = [ float(x.rstrip()) for x in f.readlines() ]
    return np.array(data)

def getData(data_folder,leg_group,stat_type):
    path_to_file = os.path.join(data_folder, leg_group + '_' + stat_type + '.csv')
    data = fileToArray(path_to_file)
    return data
stat_types = ['swing_times','swing_times','gait_cycles','duty_factors']

leg_group = 'legs1-3'

data_folder_one = '3iy_control-cw-2min-iw'
data_folder_two = '3iy_1mg_cw_6min-iw'
plotTitle = '1 mg/mL 3-IY'
keyword = '3iy'

for stat_type in stat_types:
    data_one = getData(data_folder_one,leg_group,stat_type)
    data_two = getData(data_folder_two,leg_group,stat_type)

    comp = [data_one,data_two]
    yLabelText = stat_type.replace('_',' ')
    plotLabels = [data_folder_one,data_folder_two]

    f,ax = plotData(comp,plotTitle,yLabelText,plotLabels,'t')
    figname = keyword + '_' + leg_group + '_' + stat_type + '.png'
    plt.savefig(figname)