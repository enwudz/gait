#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:30:53 2022

@author: iwoods

usage: python initializeClip.py movie_file.mov 
    ... if no movie file provided it will ask you which .mov to choose
    
This program will:
    make an excel file with same name as the video clip, with .xlsx extension

It tries to extract info from the filename
    ... and adds the info to the 'identity' sheet
    ... and reports the info
    ... asks if looks good, invites user to go in and fix/add more

"""
import sys
import gaitFunctions
import pandas as pd
import numpy as np
import re

def main(movie_file, printme = True):
    
    # is there already an excel file for this clip?
    excel_file_exists, excel_filename = gaitFunctions.check_for_excel(movie_file)
    
    needFrames = False
        
    # if there is a file, extract the info from the identity sheet
    if excel_file_exists:
        print('... found an excel file for this clip!')
        df = pd.read_excel(excel_filename, sheet_name='identity', index_col=None)
        info = dict(zip(df['Parameter'].values, df['Value'].values))
        
        if '#frames' not in info.keys():
            needFrames = True
        if 'species' not in info.keys():
            info = extract_info(movie_file)
            needFrames = True

    # if there is no file ... guess info from the filestem, and make a file!
    else:      
        
        print('... no file yet - guessing info from file stem')
        info = extract_info(movie_file)
        print('... making an excel file: ' + excel_filename)
        df = pd.DataFrame([info])
        with pd.ExcelWriter(excel_filename) as writer:
            df.to_excel(writer, index=False, sheet_name='identity')
        needFrames = True
    
    if needFrames:
        # get info for movie file
        vid_width, vid_height, vid_fps, vid_frames, vid_length = gaitFunctions.getVideoData(movie_file, False)
        info['width'] = vid_width
        info['height'] = vid_height
        info['fps'] = vid_fps
        info['duration'] = vid_length
        
        # get and save frame times for movie if not already there
        frame_times = gaitFunctions.getFrameTimes(movie_file)
        info['#frames'] = len(frame_times)
        
        # save first and last frames if not already there
        first_frame, last_frame = gaitFunctions.getFirstLastFrames(movie_file)
        gaitFunctions.saveFirstLastFrames(movie_file, first_frame, last_frame)
        
        make_identity_sheet(excel_filename, info)  
    
    if printme:
        # print the info we have, and invite user to modify the file
        print_order = gaitFunctions.identity_print_order()
        
        print('\nHere is info we have:')
        printed = []
        for thing in print_order:
            if thing in info.keys():
                print(' ' + thing + ': ' + str(info[thing]))
            else:
                print(' ' + thing + ': unknown')
            printed.append(thing)
            
        # what if there are things in the excel file that are not in print_order?
        for k in info.keys():
            if k not in printed:
                print(' ' + k  + ': ' + str(info[k]))
        print('\n If any of that needs to be changed, feel free to edit ' + excel_filename + '\n')

    return info

def make_identity_sheet(excel_filename, info):
    print_order = gaitFunctions.identity_print_order()
    vals = [info[x] for x in print_order if x in print_order]
    d = {'Parameter':print_order,'Value':vals}
    df = pd.DataFrame(d)
    with pd.ExcelWriter(excel_filename, if_sheet_exists='replace', engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, index=False, sheet_name='identity')


def guessTreatment(s):
    
    s = s.lower()
    
    drugs = ['control', 'caffeine', 'alcohol', 'nicotine', 'disulfiram', 'lead', 'simvastatin']
    conditions = ['control', 'wildtype', 'embryo']
    
    treatment = 'unknown'
    
    for drug in drugs:
        if drug in s:
            treatment = drug
            
    for condition in conditions:
        if condition in s:
            treatment = condition
            
    if 'day' in s:
        daypos = s.find('day')
        daynumber = s[daypos+3:daypos+4]
        if len(re.findall('[0-9]', daynumber)) > 0:
            treatment = 'day' + daynumber
            
    return treatment

def guessIdentity(s):
    
    s = s.lower()
    initials = 'unknown'
    
    if '_' in s:
        things = s.split('_')
        for thing in things:
            if len(thing) == 2 and len(re.findall('[a-z]{2}', thing)) > 0:
                return thing
    return initials
           
def guessSpecies(s):
    
    s = s.lower()
    
    critters = ['human','tardigrade','cat','dog','insect','tetrapod','hexapod']
    num_legs = [2, 8, 4, 4, 6, 4, 6]
    individual_number = 'unknown'
    
    species_legs = dict(zip(critters,num_legs))
    
    species = 'unknown'
    individual_number = 'unknown'
    legs = 8
    
    for critter in critters:
        if critter in s:
            species = critter
            
            # get the individual number
            if '#' in s:
                s=s.replace('#','')
            
            havecritternumber = False
            for i in np.arange(3,0,-1):
                if havecritternumber == False:
                    numsearch = critter + '[0-9]' * i
                    print(numsearch)
                    numfound = re.findall(numsearch,s)
                    if len(numfound) > 0:
                        individual_number = numfound[0].replace(critter,'')
                        havecritternumber = True
                      
            # get the number of legs
            legs = species_legs[critter]
            
    return species, legs, individual_number
    
def guessTimeRange(s):
    
    if '-' in s:
        for i in np.arange(5,0,-1):
            numsearch = '[0-9]' * i + '-' + '[0-9]' * i
            foundnum = re.findall(numsearch, s)
            if len(foundnum) > 0:
                return foundnum[0]
    else:
        return 'unknown'
    
def guessDate(s):
    mon = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    fullmonths = ['january','february','march','april','may','june','july','august','september','october','december']
    allmonths = fullmonths + mon 

    s = s.lower()
    
    month = 'unknown'
    for m in allmonths:
        date = 'unknown'
        if m in s:
            month = m
            for i in np.arange(2,0,-1):
                datesearch = '[0-9]' * i + m
                founddate = re.findall(datesearch, s)
                if len(founddate) > 0:
                    date = founddate[0].replace(m,'')
                    return month, date
                else:
                    datesearch = m + '[0-9]' * i
                    founddate = re.findall(datesearch, s)
                    if len(founddate) > 0:
                        date = founddate[0].replace(m,'')
                        return month, date
    return month, date
    
def extract_info(movie_file):
    file_stem = movie_file.split('.')[0]
    info = {}
    
    treatment = guessTreatment(file_stem)
    initials = guessIdentity(file_stem)
    species, legs, individual_number = guessSpecies(file_stem)
    timerange = guessTimeRange(file_stem)
    month, date = guessDate(file_stem)
    
    info['month'] = month
    info['date'] = date
    info['treatment'] = treatment
    info['initials'] = initials
    info['individualID'] = individual_number
    info['time_range'] = timerange
    info['file_stem'] = file_stem
    info['species'] = species
    info['num_legs'] = legs
    
    return info
    
    
    
if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
        movie_file = gaitFunctions.selectFile(['mp4','mov'])
     
    print(movie_file)    
    
    if '.mov' in movie_file or '.mp4' in movie_file:
        main(movie_file)
    else:
        exit('No movie file found')
        
        
        