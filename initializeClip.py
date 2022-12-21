#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:30:53 2022

@author: iwoods

usage: python initializeClip.py movie_file.mov 
    ... if no movie file provided it will ask you which .mov to choose
    
This program will:
    make an excel file with 5 tabs: 
            identity
            pathtracking
            path_stats
            steptracking
            step_stats
    this excel has same name as the video clip, with .xlsx extension

It tries to extract info from the filename
    ... and adds the info to the 'identity' sheet
    ... and reports the info
    ... asks if looks good, invites user to go in and fix/add more

"""
import sys
import gaitFunctions
import pandas as pd
import re

def main(mov_file):
    
    # is there already an excel file for this clip?
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(mov_file)
        
    # if there is a file, extract the info from the identity sheet
    if excel_file_exists:
        print('... found an excel file for this clip!')
        df = pd.read_excel(excel_filename, sheet_name='identity', index_col=None)
        info = dict(zip(df['Parameter'].values, df['Value'].values))
    
    # if there is no file ... guess info from the filestem, and make a file!
    else:      
        print('... no file yet - guessing info from file stem')
        info = extract_info(mov_file)
        print('... making an excel file: ' + excel_filename)
        make_excel(excel_filename, info)
    
    # print the info we have, and invite user to modify the file
    print_order = gaitFunctions.identity_print_order()
    
    print('\nHere is info we have - feel free to edit ' + excel_filename + '\n')
    printed = []
    for thing in print_order:
        print(' ' + thing + ': ' + info[thing])
        printed.append(thing)
        
    # what if there are things in the excel file that are not in print_order?
    for k in info.keys():
        if k not in printed:
            print(' ' + k  + ': ' + str(info[k]))
    
    print('\n')
    return info

def make_excel(excel_filename, info):
    print_order = gaitFunctions.identity_print_order()
    vals = [info[x] for x in print_order]
    d = {'Parameter':print_order,'Value':vals}
    df = pd.DataFrame(d)
    df2 = pd.DataFrame()
    with pd.ExcelWriter(excel_filename) as writer:
        df.to_excel(writer, index=False, sheet_name='identity')
        df2.to_excel(writer, index=False, sheet_name='pathtracking')
        df2.to_excel(writer, index=False, sheet_name='path_stats')
        df2.to_excel(writer, index=False, sheet_name='steptracking')
        df2.to_excel(writer, index=False, sheet_name='step_timing')
        df2.to_excel(writer, index=False, sheet_name='step_stats')
        df2.to_excel(writer, index=False, sheet_name='gait_styles')

def guess_the_thing(thing):
    ''' what is this thing?
    choices are: initials, date, treatment, individualID, time_range '''
    
    month_abbreviations = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    month_names = ['january','february','march','april','may','june','july','august','september','october','november','december']
    all_months = month_abbreviations + month_names
    
    # is the thing initials?
    if len(thing) == 2 or len(thing) == 3:
        if thing == 'wt':
            return 'treatment'
        else:
            return 'initials'
    
    elif '-' in thing:
        return 'time_range'
    
    # is the thing a treatment?
    elif 'control' in thing or 'wildtype' in thing:
        return 'treatment'
    
    # is the thing an individual?
    elif 'tardigrade' in thing or 'sample' in thing:
        return 'individualID'
    
    else:
        # is the thing a date?
        # are there any numbers in the thing? If so, remove them
        checked = re.findall('[0-9]+', thing)
        if len(checked) > 0:
            for num in checked:
                thing = thing.replace(num,'').lower()
            if thing.lower() in all_months:
                return 'date'
        else:
            return 'treatment'
    
def extract_info(mov_file):
    
    file_stem = mov_file.split('.')[0]
    info = {}
    info['date'] = ''
    info['treatment'] = ''
    info['initials'] = ''
    info['individualID'] = ''
    info['time_range'] = ''
    info['file_stem'] = file_stem
    
    stuff = file_stem.split('_')
    if len(stuff) > 0:
        for thing in stuff:
            best_guess = guess_the_thing(thing)
            if best_guess in info.keys():
                if len(info[best_guess]) > 0:
                    info[best_guess] += '_' + thing
                else:
                    info[best_guess] = thing
                    
    return info
    
if __name__== "__main__":

    if len(sys.argv) > 1:
        mov_file = sys.argv[1]
    else:
        mov_file = gaitFunctions.select_movie_file()
        
    if '.mov' in mov_file:
        main(mov_file)
    else:
        exit('No .mov file found')