#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:58:34 2023

@author: iwoods

Opens an excel file (output of combineClips.py)

Checks what kind of data is available, in these four sheets
    path_summaries - path tracking data from trackCritter.py and analyzeTrack.py
    step_timing - all data from frameStepper.py and analyzeSteps.py
    step_summaries - step parameters (from frameStepper.py and analyzeSteps.py)
    gait_summaries - gait styles (from frameStepper.py and analyzeSteps.py)
    
Based on available data, offers options for comparisons
    path_summaries -  pick groups (assume want to compare treatments)
    step_timing -     pick an individual or a treatment and show options (borrowed from from plotClip.py)
    step_summaries -  pick groups (assume want to compare treatments)
    gait_summaries -  pick groups (assume want to compare treatments)

Based on selected comparison, offers options for plots
    show columns in selected sheets, and select
    (in step_summaries, see plotClip.py)

Plots!

"""
import sys
import gaitFunctions
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main(datafile):

    # open the combined data excel file (from combineClips.py) and check what kind of data is available
    data_dfs, data_descriptions = getSheets(datafile)
    if len(data_dfs.keys()) == 1:
        selected_dataset = list(data_dfs.keys())[0]
    elif len(data_dfs.keys()) > 1:   
        datatype_options = [datatype + ': ' + data_descriptions[datatype] for i, datatype in enumerate(data_dfs.keys())]
        selection = gaitFunctions.selectOneFromList(datatype_options)
        selected_dataset = selection.split(':')[0] # klugey
    else:
        print('\n  No data available in ' + datafile + '!\n')

    # get analysis and plot options for the selected dataset
    if selected_dataset == 'path_summaries':
        summaryPlots(data_dfs['path_summaries'], 5)
    if selected_dataset == 'step_timing':
        stepTimingPlots(data_dfs['step_timing'])
    if selected_dataset == 'step_summaries':
        summaryPlots(data_dfs['step_summaries'], 3)
    if selected_dataset == 'gait_summaries':
        summaryPlots(data_dfs['gait_summaries'], 5)

    # while plotting, show plot options. When finished, exit.    

def stepTimingPlots(step_df):
    '''
    Note: for step_timing data, can use these (just need f = figure, and step_timing dataframe)
        gaitFunctions.stepParameterPlot(f, stepdata_df) OK
        gaitFunctions.stepParameterLeftRightPlot(f, stepdata_df) OK
        gaitFunctions.speedStepParameterPlot(f, stepdata_df) OK
        gaitFunctions.swingOffsetPlot(f, stepdata_df) OK
        gaitFunctions.metachronalLagLRPlot(f, stepdata_df) OK
    '''
    
    # what individuals are present in the data set?
    uniq_ids = np.sort(np.unique(step_df['uniq_id'].values))
    
    # select an individual or multiple individuals
    print('Which individual(s) would you like to see?')
    selection = gaitFunctions.selectMultipleFromList(uniq_ids)
    selected_df = step_df[step_df['uniq_id'].isin(selection)]
    
    # offer plot options
    plotting = True
    while plotting:
        selected_plot = selectStepPlot()
        
        if selected_plot == 'finished':
            plotting = False
            
        elif selected_plot == 'step parameters':
            f, axes = plt.subplots(1,5, figsize=(14,3), constrained_layout=True)
            f = gaitFunctions.stepParameterPlot(f, selected_df)
            plotMessage(selected_plot)
            plt.show()
            
        elif selected_plot == 'left vs. right':
            f, axes = plt.subplots(1,5, figsize = (14,3), constrained_layout=True)
            f = gaitFunctions.stepParameterLeftRightPlot(f, selected_df)
            plotMessage(selected_plot)
            plt.show()
            
        elif selected_plot == 'speed vs. steps':
            f, axes = plt.subplots(1,5, figsize = (14,3), constrained_layout=True)
            f = gaitFunctions.speedStepParameterPlot(f, selected_df)
            plotMessage(selected_plot)
            plt.show()  
            
        elif selected_plot == 'swing offsets':
            f, axes = plt.subplots(2,3, figsize = (10,6), constrained_layout=True)
            f = gaitFunctions.swingOffsetPlot(f, selected_df)
            plotMessage(selected_plot)
            plt.show()  
            
        elif selected_plot == 'metachronal lag':
            f, axes = plt.subplots(1,2, figsize = (8,3), constrained_layout=True)
            f = gaitFunctions.metachronalLagLRPlot(f, selected_df)
            plotMessage(selected_plot)
            plt.show()
            
        else:
            print('Invalid plot selection, try again...')
    

def plotMessage(selected_plot):
    print(' ... showing ' + selected_plot + ', close window to proceed')

def selectStepPlot():
    print('\nPlot options: \n')
    print('  0. finished = quit plotting')
    
    option_list = ['step parameters',
                   'left vs. right',
                   'speed vs. steps',
                   'swing offsets',
                   'metachronal lag']
    
    option_descriptions = ['show step parameters (stance, swing, duty factor, cycle, distance)',
                           'show step parameters comparing left vs. right lateral legs',
                           'show scatter plot of speed vs step parameters (for lateral legs)',
                           'show swing-swing timing offsets',
                           'show elapsed time between 3rd leg swing and 1st leg swing']
    
    for i in np.arange(len(option_list)):
        print('  ' + str(i+1) + '. ' + option_list[i] + ':  ' + option_descriptions[i])
    
    selection = input('\nChoose one: ')
    
    try:
        ind = int(selection) - 1
        if ind == -1:
            to_plot = 'finished'
            print('... Finished Plotting!\n')
        else:
            to_plot = option_list[ind]
            print('You chose ' + to_plot)
    except:
        print('\ninvalid selection, choosing ' + option_list[0])
        to_plot = option_list[0]
        
    return to_plot
    
def summaryPlots(df, start_data_col):
    
    # print(df.head(3)) # testing
    plot_options = df.columns.values[start_data_col:] # which columns have data points?

    # what treatments are represented in this dataset?
    treatments = np.unique(df['treatment'])
    
    # if num treatments > 1, assume that we want to compare treatments
    if len(treatments) > 1:
        groups = [[x] for x in sorted(treatments)]
        groupcol = 'treatment'
        print('We will compare ' + ' vs. '.join(sorted(treatments)))
        groupnames = [g[0] for g in groups]
    
    # if num treatmens == 1, assume that we want to select individuals to compare
    elif len(treatments) == 1:
        # make groups of individuals to compare from within this treatment      
        groupcol = 'individual'
        individuals = np.unique(np.sort(df[groupcol].values))
        print(individuals)
        
        selecting = True
        while selecting: 
            print('\nJust one treatment here')
            print(' ... assume we want to compare individuals or groups of individuals\n')
            
            selection = input('How many groups should we make? Enter an integer: ')
            
            try:
                num_groups = int(selection)
                selecting = False
            except:
                print('\n ... invalid selection, try again ...')
        
        groups = []
        for g in np.arange(num_groups):
            print('\nSelect individual(s) for group ' + str(g+1) + ' of ' + str(num_groups))
            selection = gaitFunctions.selectMultipleFromList(individuals)
            groups.append(selection)
            
        groupnames = ['\n'.join(x) for x in groups]
    
    # Offer options to plot, and do the plots
    plotting = True
    while plotting:
        
        selection = selectPlotOption(plot_options)
        
        if selection == 'finished':
            plotting = False
        
        else:
            
            print('\nHere is the comparison of ' + selection)
            print('Close plot window to continue')
            
            # plot the selection
            makeBoxPlot(df, groupcol, groupnames, groups, selection)

def makeBoxPlot(df,col,groupnames,groups,datacol):

    f,ax = plt.subplots(figsize=(3,4))    

    if 'control' in groupnames:
        
        print('Rearranging group so control is first!')
        ind = groupnames.index('control')
        control = groupnames.pop(ind)
        controldata = groups.pop(ind)
        groupnames.insert(0, control)
        groups.insert(0,controldata)
    
    # collect data
    data_to_plot = []
    for i,group in enumerate(groups):
        # print(group, col, datacol) # testing
        data_to_plot.append(df[df[col].isin(group)][datacol].values)
    # print(data_to_plot) # testing
    
    # make boxplot
    bp = plt.boxplot(data_to_plot, patch_artist=True, showfliers=False)
    # # bp = gaitFunctions.formatBoxPlots(bp, ['tab:blue'], ['white'], ['lightsteelblue']) # boxcolor, mediancolors, fliercolors
    bp = gaitFunctions.formatBoxPlots(bp, [[0,0,0.384]] , [[ 0.76, 0.86, 0.85 ]],  ['lightsteelblue'])
    
    # add scatter over the boxplot
    a = 1 # alpha
    sc = 'k' # [ 0.76, 0.86, 0.85 ] # 'k' # color
    sz = 30 # marker size
    ji = 0.05 # jitter around midline
    for i, group in enumerate(groups):   
        print(data_to_plot[i])
        xScatter = np.random.normal(i+1, ji, size=len(data_to_plot[i]))
        print(xScatter)
        ax.scatter(xScatter, data_to_plot[i], s=sz, c=sc, alpha = a)
    
    # do some stats?
    if len(data_to_plot) == 2:
        gaitFunctions.statsFromBoxData(data_to_plot, 'kw')
    
    # add axes labels
    plt.ylabel(datacol, fontsize=12)
    plt.xticks(np.arange(len(groups))+1,groupnames)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_facecolor("lightgray")
    plt.subplots_adjust(left = 0.3)
    
    plt.show()
    

def selectPlotOption(option_list):
    print('\nPlot options: \n')
    print('  0. finished = quit plotting')
    
    for i in np.arange(len(option_list)):
        print('  ' + str(i+1) + '. ' + option_list[i] )
    
    selection = input('\nChoose one: ')
    
    try:
        ind = int(selection) - 1
        if ind == -1:
            to_plot = 'finished'
            print('... Finished Plotting!\n')
        else:
            to_plot = option_list[ind]
            print('You chose ' + to_plot)
    except:
        print('\ninvalid selection, choosing ' + option_list[0])
        to_plot = option_list[0]
        
    return to_plot
    
    

def getSheets(excel_file):
    
    all_datatypes = np.array(['path_summaries', 'step_timing', 'step_summaries', 'gait_summaries'])
    
    all_descriptions = np.array(['path tracking data from trackCritter.py and analyzeTrack.py',
                         '   individual step data from frameStepper.py and analyzeSteps.py',
                         'step parameter averages from frameStepper.py and analyzeSteps.py',
                         'gait style averages from frameStepper.py and analyzeSteps.py'])

    all_dataframes = np.empty(4, dtype = object)
    
    for i, datatype in enumerate(all_datatypes):
        try:
            all_dataframes[i] = pd.read_excel(excel_file, sheet_name=datatype, index_col=None)
        except:
            all_dataframes[i] = pd.DataFrame()

    has_data = np.array([len(x) for x in all_dataframes])
    
    
    datatypes = all_datatypes[np.where(has_data > 0 )]
    dataframes = all_dataframes[np.where(has_data > 0 )]
    descriptions = all_descriptions[np.where(has_data > 0)]
    
    data_dfs = dict(zip(datatypes, dataframes))
    data_descriptions = dict(zip(datatypes, descriptions))
    

    return data_dfs, data_descriptions
    

if __name__== "__main__":

    if len(sys.argv) > 1:
        
        datafile = sys.argv[1]
        
        if '.xlsx' not in datafile:
            print('Please choose an .xlsx file to analyze')
        else:
            main(datafile)
    
    else:
        
        datafiles = sorted(glob.glob('*combined.xlsx'))
        
        if len(datafiles) == 1:
            datafile = datafiles[0]
            print('\n ... opening ' + datafile + ' ...\n')
            main(datafile)
            
        elif len(datafiles) > 0:
            datafile = gaitFunctions.selectOneFromList(datafiles)
            main(datafile)
        
        else:
            print('Cannot find any combined data files in this directory ...')
            