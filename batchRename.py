#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:43:48 2023

@author: iwoods
"""

import os
import glob
import pandas as pd

replaceme = 'tardigrade'
newthing = 'tardigrade0'

movs = glob.glob('*.mov')
mp4s = glob.glob('*.mp4')

pngs = glob.glob('*.png')

moviefiles = movs + mp4s

# rename the png files
# print('\nRenaming png images')
# for f in pngs:
#     if replaceme in f:
#         newf = f.replace(replaceme,newthing)
#         print('\t', f, newf)
#         os.rename(f,newf)

# rename the movie files
print('\nRenaming movie files')
for f in moviefiles:
    if replaceme in f:
        newf = f.replace(replaceme,newthing)
        print('\t', f, newf)
        os.rename(f,newf)

# rename the excel files and update the identity sheet
print('\nUpdating identity sheets)')
xls  = sorted(glob.glob('*.xlsx'))
for x in xls:
    
    if replaceme in x:
    
        print(' ... Updating ' + x)
        
        info_df = pd.read_excel(x, sheet_name='identity', index_col=None)
        
        parameters = info_df['Parameter'].values
        values = info_df['Value'].values
    
        oldfstem = values[0]
        newfstem = oldfstem.replace(replaceme,newthing)
    
        values[0] = newfstem
        
        d = {'Parameter':parameters,'Value':values}
        df = pd.DataFrame(d)
        with pd.ExcelWriter(x, if_sheet_exists='replace', engine='openpyxl', mode='a') as writer:
            df.to_excel(writer, index=False, sheet_name='identity')
    
        os.rename(x, x.replace(replaceme, newthing))
    
