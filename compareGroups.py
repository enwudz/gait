#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:53:38 2023

@author: iwoods
"""

import glob
import gaitFunctions
import pandas as pd
import numpy as np
import sys

"""
WISH LIST

save a COMPARISON file (text?)
    provide as part of command line
    OR list available comparisons
    IF no comparison, then get the groups

"""

def main(group_file = ''):

    ## ===> select groups to compare

    # groups = a dictionary to hold list of clips for each group
    # key = group name
    # value = list of clips in that group
    groups = getGroups(group_file)
    
    # Print out the groups (comment out?)
    printGroups(groups)

    ## ===> load and combine data by group


    ## ===> offer options to plot

    plotting = True
    while plotting:

        plotting = False
        if plotting == False:
            break

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

def addToGroup(groups, category_dict, category_type):

    for group in sorted(groups.keys()):
        
        # if there's only one member of this category, we do not need to make a selection
        if len(category_dict) == 1: 
            return groups
            
        # if there is more than one member of this category, select what we want in the group
        elif len(category_dict) > 1:
            categories = list(sorted(category_dict.keys()))
            print('\nWhat ' + category_type.upper() + '(s) should be in the group ' + group + '?')

            selections = selectFromList(categories)

            if selections == 'all':
                for k in categories:
                    groups[group].extend(category_dict[k])
            else:
                for selection in selections:
                    groups[group].extend(category_dict[selection])

            # make sure only unique items in the group list
            groups[group] = list(set(sorted(groups[group])))
    
    return groups

def selectFromList(li):

    choice = []
    i = 1
    for thing in li:
        print('   ' + str(i) + ': ' + thing)
        i += 1
    print('   ' + str(i) + ': ALL of these')

    entry = input('\nSelect from the above options, separated by SPACES: ').rstrip()

    try:
        selected_numbers = [int(x) for x in entry.split(' ')]
    except:
        print(entry + ' is not a valid selection, choosing them all')
        choice = 'all'
        return choice

    for num in selected_numbers:
        if num - 1 == len(li):
            print('Choosing them all')
            choice = 'all'
        elif num > len(li):
            print(str(num) + ' is too big - choosing them all')
            choice = 'all'
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
            
    for group in group_names:
        groups[group] = []
        
    # Put the available data into categories
    dates, treatments, individuals, collectors = categorizeClips()

    # Select the clips that should be in the group(s)
    categories = ['treatment','date','individual','collector']
    category_dicts = [treatments, dates, individuals, collectors]
    for i, category in enumerate(categories):
        groups = addToGroup(groups, category_dicts[i], category)
        
    # Save groups (comment out?)
    saveGroups(groups)

    return groups
        

def categorizeClips():

    print(' ... putting the clips into different categories ... ')
    
    # make empty dictionaries to hold lists of clips
    dates = {}
    treatments = {}
    individuals = {}
    collectors = {}

    # go through all data (excel) files and add them to the appropriate dictionaries
    excel_files = sorted(glob.glob('*.xlsx'))

    for file in excel_files:
        # open file and get data from the identity tab
        try:
            df = pd.read_excel(file, sheet_name = 'identity', index_col=None)
            # print('reading data from ' + file)
        except:
            print('No identity info available in ' + file)
            next
        info = dict(zip(df['Parameter'].values, df['Value'].values))

        for category in info.keys():
            
            if category == 'treatment':
                treatment = info['treatment']
                if treatment in treatments.keys():
                    treatments[treatment].append(file)
                else:
                    treatments[treatment] = [file]
                    
            if category == 'date':
                date = info['date']
                if date in dates.keys():
                    dates[date].append(file)
                else:
                    dates[date] = [file]
                    
            if category == 'individualID':
                individual = info['individualID']
                if individual in individuals.keys():
                    individuals[individual].append(file)
                else:
                    individuals[individual] = [file]

            if category == 'initials':
                collector = info['initials']
                if collector in collectors.keys():
                    collectors[collector].append(file)
                else:
                        collectors[collector] = [file]

    return dates, treatments, individuals, collectors


def checkForSavedGroups():
    
    group_list = glob.glob('*compare.txt')
    if len(group_list) > 0:
        
        print('\nChoose from this list : ')
        i = 1
        li = sorted(group_list)
        
        for thing in li:
            print(str(i) + ': ' + thing)
            i += 1
            
        print(str(i) + ': make new groups to compare')
        
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
        group_file = checkForSavedGroups()
        
    main(group_file)
        
