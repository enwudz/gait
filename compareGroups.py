#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:53:38 2023

@author: iwoods
"""

import glob
import gaitFunctions
import pandas as pd
# import numpy as np
import sys

"""
WISH LIST

what if there is no stepdata or gaitdata or trackdata?

WORKING ON THIS!!
save group data ... and when loading data, check to see if it exists already.

"""

def main(group_file = ''):

    ## ===> get or select GROUPS to compare
    # groups = a dictionary to hold list of clips for each group
    # key = group name
    # value = list of clips in that group
    if len(group_file) == 0:
        group_file = checkForSavedGroups()
    groups = getGroups(group_file)
    
    # Print out the groups (comment out if do not want to see)
    # printGroups(groups)

    ## ===> load and combine data by group
    tracking_dfs, stepdata_dfs, gaitdata_dfs = loadData(groups)

    print(tracking_dfs[1].head(10))

    ## ===> offer options to plot
    
    # these depend on how many groups we have
    # and what kind of data we have (tracked path, step timing, gait styles)
    # see step_data_plots.ipynb for some ONE GROUP plots . . .
    # see compare_step_parameters for some multiple group plots

    plotting = True
    while plotting:

        plotting = False
        if plotting == False:
            break

def getSavedData(groupname):
    
    saved_datafile = groupname + '_data.xlsx'
    
    tracked_df = pd.DataFrame()
    stepdata_df = pd.DataFrame()
    gaitdata_df = pd.DataFrame()
    
    if len(glob.glob(saved_datafile)) == 0:
        return tracked_df, stepdata_df, gaitdata_df
    

    
    return tracked_df, stepdata_df, gaitdata_df

def loadData(groups):
    # Make lists to collect dataframes from each group
    tracking_dfs = [] # = from pathtracking tab
    stepdata_dfs = [] # = from step_timing tab
    gaitdata_dfs = [] # = from gait_styles tab

    # for each group ... go through list of clips in the group
    # and add data from that clip to existing dataframe

    for group in groups.keys():
        
        print(' ... loading data for ' + group + ' ... ')
        
        # check to see if there is data already saved for this group
        tracked_df, stepdata_df, gaitdata_df = getSavedData(group)
        
        if len(tracked_df) == 0:

            # go through each clip in this group
            clips = groups[group]
            for i, clip in enumerate(clips):
                
                
                # if loading first clip, that becomes the dataframe to store all clips
                # this means that the first clip MUST have all the data! hmmm....
                if i == 0:
                    
                    # get pathtrackng data and add a column for the clip name
                    tracked_df, excel_filename = gaitFunctions.loadTrackedPath(clip)
                    tracked_df = addCliptoDF(tracked_df, clip)
                    
                    # get step_timing data and add a column for the clip name
                    stepdata_df = gaitFunctions.loadStepData(clip, excel_filename)
                    stepdata_df = addCliptoDF(stepdata_df, clip)
                    
                    # get step_timing data and add a column for the clip name
                    gaitdata_df = gaitFunctions.loadGaitData(clip, excel_filename)
                    gaitdata_df = addCliptoDF(gaitdata_df, clip)
                
                # when loading subsequent clips
                #     load dataframe and add a column for clip name
                #     add to existing dataframe
                else:
                    
                    # get pathtrackng data and add a column for the clip name
                    tdf, excel_filename = gaitFunctions.loadTrackedPath(clip)              
                    if len(tdf) > 0:
                        tdf = addCliptoDF(tdf, clip)
                    else:
                        print(' ... no path tracking data available for ' + clip)
                    
                    # get step_timing data and add a column for the clip name
                    sdf = gaitFunctions.loadStepData(clip, excel_filename)
                    if len(sdf) > 0:
                        sdf = addCliptoDF(sdf, clip)
                    else:
                        print(' ... no step timing data available for ' + clip)
                    sdf = addCliptoDF(sdf, clip)
                    
                    # get step_timing data and add a column for the clip name
                    gdf = gaitFunctions.loadGaitData(clip, excel_filename)
                    if len(gdf) > 0:
                        gdf = addCliptoDF(gdf, clip)
                    else:
                        print(' ... no step gait style data available for ' + clip)
                    gdf = addCliptoDF(gdf, clip)
                    
                    # concatenate new clips:  existing_df = pd.concat([existing_df, new_df])
                    tracked_df = pd.concat([tracked_df, tdf])
                    stepdata_df = pd.concat([stepdata_df, sdf])
                    gaitdata_df = pd.concat([gaitdata_df, gdf])               
                
        # done collecting data for this group!
        
        # save data for this group
        saved_datafile = group + '_data.xlsx'
        print(saved_datafile)
        
        # add data for this group to the list of dataframes
        tracking_dfs.append(tracked_df)
        stepdata_dfs.append(stepdata_df)      
        gaitdata_dfs.append(gaitdata_df)
        
    return tracking_dfs, stepdata_dfs, gaitdata_dfs


def addCliptoDF(df, clip):
    clipname = clip.split('.')[0]
    num_rows = df.shape[0]
    clip_column = [clipname] * num_rows
    df['clip'] = clip_column
    return df

def loadGroups(group_file):
    
    groups = {}
    
    with open(group_file,'r') as f:
        for line in f:
            line = line.rstrip()
            if 'group:' in line:
                category = line.split(': ')[1]
                groups[category] = []
            elif len(line) > 0:
                groups[category].append(line)    
                
    return groups

def saveGroups(groups):
    
    print('\nBriefly (~8-15 characters?) describe these groups (no spaces or periods)')
    comparison_name = input('   (this description will be used as a saved file name):  ')
    group_file = comparison_name + '_compare.txt'
    print(' ... Saving ' + group_file + ' ...\n')
    o = open(group_file, 'w')
    for group in sorted(groups.keys()):
        o.write('\ngroup: ' + group + '\n')
        o.write('\n'.join(sorted(groups[group])))
        o.write('\n')
    o.close()
    
    return

def printGroups(groups):
    for group in sorted(groups.keys()):
        print('\n' + group + ' group:')
        print('\n'.join(sorted(groups[group])))
        print('\n')
    return


def selectFromList(li, category = ''):
    
    if len(li) == 1: # no need to choose, just one thing
        return li
    
    choice = []
    i = 1
    
    if len(category) > 0:
        print('\nWhich ' + category + ' should we choose?')
        
    for thing in li:
        print('   ' + str(i) + ': ' + thing)
        i += 1
    print('   ' + str(i) + ': ALL of these')

    entry = input('\nSelect from the above options, separated by SPACES: ').rstrip()

    try:
        selected_numbers = [int(x) for x in entry.split(' ')]
    except:
        print(entry + ' is not a valid selection, choosing them all')
        return li

    for num in selected_numbers:
        if num - 1 == len(li):
            print('Choosing them all')
            choice = li
        elif num > len(li):
            print(str(num) + ' is too big - choosing them all')
            choice = li
        else:
            ind = int(num) - 1
            choice.append(li[ind])
    
    return choice
            
def getGroups(group_file = ''):

    groups = {}
    
    # if group_file provided, load it in!
    if len(group_file) > 0:
        groups = loadGroups(group_file)
        return groups

    # if no movie_data file provided ... build up a group or groups to compare
    selection = input('\nEnter number of groups: ').rstrip()
    try:
        num_groups = int(selection)
    except:
        print('\nInvalid entry, setting number of groups to 1!')
        num_groups = 1

    group_names = []
    for i in range(num_groups):
        group_number = str(i + 1)
        default_name = 'group ' + group_number
        selection = input('\nEnter name for group ' + group_number + ' (default = ' + default_name + '): ').rstrip()
        if len(selection) == 0:
            group_names.append(default_name)
        else:
            group_names.append(selection)
    
    # Select the clips that should be in the group(s)
    clipinfo, categories = makeClipDict()
    for group in group_names:
        
        print('\nBUILDING GROUP: ' + group + '!')
        groups[group] = makeGroup(clipinfo, categories)
        
    # Save groups (comment out?)
    saveGroups(groups)

    return groups

def makeGroup(clipinfo, categories):
    
    cliplist = sorted(clipinfo.keys())
    
    for category in categories:
        
        # which options are available in this category?
        options = []
        for clip in cliplist:
            if clipinfo[clip][category] not in options:
                options.append(clipinfo[clip][category])
        
        # select which ones to keep
        selections = selectFromList(options, category)
        
        # update cliplist with the 'kept' clips
        # can probably do this with list comprehension . . .
        keepers = []
        for option in selections: # different options within a category
            for clip in cliplist: 
                if clipinfo[clip][category] == option:
                    keepers.append(clip)
        
        cliplist = keepers
    
    return cliplist

        
def makeClipDict():
    '''
    Parameters
    -------
    None ... though there need to be .xlsx files from tracking experiments
    in the same folder

    Returns
    -------
    clips : dictionary
        keys = clip names
        values = dictionaries of categories
        e.g. clips[clip name]['treatment'] = 'drug'
        or   clips[clip name]['date'] =      '8 Dec'
        
    categories : list
        list of categories (keys)

    '''
    
    print(' ... Getting experiment categories for clips ...')
    
    # make an empty dictionary
    clips = {}
    categories = ['treatment','date','initials','individualID'] # ,'time_range']
    
    # go through all data (excel) files
    excel_files = sorted(glob.glob('*.xlsx'))

    for file in excel_files:
        clips[file] = {}
        
        # open file and get data from the identity tab
        try:
            df = pd.read_excel(file, sheet_name = 'identity', index_col=None)
            print('reading data from ' + file)
        except:
            print('No identity info available in ' + file)
            next
        info = dict(zip(df['Parameter'].values, df['Value'].values))
        
        for category in categories:
            if category in info.keys():
                clips[file][category] = info[category]
            else:
                clips[file][category] = ''
                
    return clips, categories

def checkForSavedGroups():
    
    group_list = glob.glob('*compare.txt')
    if len(group_list) > 0:
        
        print('\nChoose from this list : ')
        i = 1
        li = sorted(group_list)
        
        for thing in li:
            print('  ' + str(i) + ': ' + thing)
            i += 1
            
        print('  ' + str(i) + ': make new groups to compare')
        
        entry = input('\nWhich ONE do you want? ')
        
        try:
            choice = int(entry)
        except:
            print(choice + ' is an invalid choice, making new groups')
            return ''
        
        if choice > len(li):
            print('... OK we will make new groups to compare! ')
            group_file = ''
        else:
            ind = choice - 1
            group_file = li[ind]
            print('\nYou chose ' + group_file + '\n')
        
    else:
        group_file = ''
    
    return group_file

if __name__== "__main__":

    if len(sys.argv) > 1:
        group_file = sys.argv[1]
    else:
        group_file = ''
        
    main(group_file)
        
