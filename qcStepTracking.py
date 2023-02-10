#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:14:59 2023

@author: iwoods

if want to compare step track timing
    add _1 to the 'old' steptracking sheets (steptracking etc) to archive them
    do step tracking again
    
compare the step data in the sheets: 
    compare step timing
        for each leg
            calculate pearson coefficient for DOWNS, for UPS, and get average
        plot each leg different color? close = down, open = up?)
        plot all legs or a selected leg
    
for each stepdata group ... 
    get the parameters for each step and compare those across . . . stance time, etc.
    
    

"""

import numpy as np
from scipy.stats.stats import pearsonr 
import matplotlib.pyplot as plt
import gaitFunctions
import sys
import pandas as pd

def main(movie_file):
    
    # scatter plot of down and up times for all legs
    mov_data_1, excel_filename = gaitFunctions.loadMovData(movie_file)
    mov_data_2, excel_filename = gaitFunctions.loadMovData(movie_file, 'steptracking_1')
    # scatterLegTimes(mov_data_1, mov_data_2)

    # compare step parameters between runs
    # lateral legs, rear legs ... so 4x2 plot of pairs
    plotLegParameterComps(excel_filename)
       
    # compare gait styles between different runs
    # lateral legs, rear legs ... so 2x1 plot of pairs

def plotLegParameterComps(excel_filename):
    
    step_timing_1 = pd.read_excel(excel_filename, sheet_name = 'step_timing', index_col=None)
    step_timing_2 = pd.read_excel(excel_filename, sheet_name = 'step_timing_1', index_col=None)
    
    # lateral_legs = gaitFunctions.get_leg_combos()[0]['lateral']
    # rear_legs = gaitFunctions.get_leg_combos()[0]['rear']
    
    parameters = ['stance','swing','gait','duty']
    leg_sets = ['lateral','rear']

    f,axes = plt.subplots(nrows=2, ncols=4, figsize=(12,6) )
    
    a = 0.4 # scatter alpha 0.5
    sc = 'slategray' # scatter color 'slategray'
    sz = 10 # scatter size 20
    ji = 0.02 # jitter 0.02

    fs = 18
    
    for l, leg_set in enumerate(leg_sets):
        for p, parameter in enumerate(parameters):
            current_ax = axes[l,p]
            legs = gaitFunctions.get_leg_combos()[0][leg_set]
            
            data_1 = step_timing_1[step_timing_1['legID'].isin(legs)][parameter].values
            data_2 = step_timing_2[step_timing_2['legID'].isin(legs)][parameter].values
            
            # boxplot with wobbly points
            bp = current_ax.boxplot([data_1,data_2], patch_artist=True, showfliers=False)
            bp = gaitFunctions.formatBoxPlots(bp, ['tab:blue'], ['white'], ['lightsteelblue'])
            
            # scatter the points
            scatter_1 = np.random.normal(1, ji, size=len(data_1))
            scatter_2 = np.random.normal(2, ji, size=len(data_2))
            current_ax.scatter(scatter_1, data_1, s=sz, c=sc, alpha = a)
            current_ax.scatter(scatter_2, data_2, s=sz, c=sc, alpha = a)
            
            # add titles
            if l == 0:
                current_ax.set_title(parameter,fontsize=fs)
                
            if p == 0:
                current_ax.set_ylabel(leg_set,fontsize=fs)

            current_ax.set_xticks([1,2], labels = ['round1', 'round2'])
    
    plt.tight_layout()
    plt.show()
            
        


def scatterLegTimes(data1, data2):
    
    f = plt.figure(figsize=(8,8))
    ax = f.add_axes([0.1,0.1,0.8,0.8])
    
    legs = np.unique(np.array([x.split('_')[0] for x in data1.keys()]))    
    plot_colors = gaitFunctions.get_plot_colors(len(legs), 'tab')
    leg_colors = dict(zip(legs,plot_colors))
    markersize=80

    for leg in legs:
        
        ups_1 = np.array([float(x) for x in data1[leg + '_up'].split()])
        downs_1 = np.array([float(x) for x in data1[leg + '_down'].split()])
        
        ups_2 = np.array([float(x) for x in data2[leg + '_up'].split()])
        downs_2 = np.array([float(x) for x in data2[leg + '_down'].split()])
        
        u1,u2 = getMatchingPoints(ups_1, ups_2)
        d1,d2 = getMatchingPoints(downs_1, downs_2)
        
        rup = np.round(pearsonr(u1,u2)[0],4)
        rdown = np.round(pearsonr(d1,d2)[0],4)
        lab = leg + ' - d: ' + str(rdown) + '; u: ' + str(rup)

        
        ax.scatter(u1,u2,s=markersize,facecolors='none',edgecolors=leg_colors[leg])
        ax.scatter(d1,d2,s=markersize,facecolors=leg_colors[leg],edgecolors=leg_colors[leg],label=lab)
    
    ax.set_xlabel('Time (sec): round 1')
    ax.set_ylabel('Time (sec): round 2')
    ax.legend()
    plt.show()
    

def getMatchingPoints(vec1,vec2):
    '''
    for each point in SHORTER vector
        find closest point in LONGER fector

    Parameters
    ----------
    vec1 : numpy array
        1-dimensional array of numbers.
    vec2 : numpy array
        1-dimensional array of numbers.

    Returns
    -------
    shortvec : numpy array
        1-dimensional array of numbers.
    matchingvec : numpy array
        1-dimensional array of numbers, same length as shortvec

    '''
    
    if len(vec1) >= len(vec2):
        longvec = vec1
        shortvec = vec2
    else:
        longvec = vec2
        shortvec = vec1
    
    matching_points = []
    for point in shortvec:
        matching_points.append(longvec[np.argmin(np.abs(longvec-point))])
        
    matchingvec = np.array(matching_points)
    
    return shortvec, matchingvec

def getSameSize(veclist):
    
    vec1, vec2 = veclist
    
    if len(vec1) == len(vec2):
        A = vec1
        B = vec2
    elif len(vec2) > len(vec1):
        A = vec1
        B = vec2[:len(vec1)]
    elif len(vec1) > len(vec2):
        A = vec1[:len(vec2)]
        B = vec2
        
    return A,B

def minDiffVectors(vec1, vec2):
    
    lowest_diff = 100
    minDiffVecs = []
    
    comps = [[vec1,vec2], # no offiset
             [vec1[1:],vec2], # shift vec1 to the right
             [vec1, vec2[1:]] # shift vec2 to the right
             ]
     
    for comp in comps:
        A,B = getSameSize(comp)
        mean_diff = np.abs(np.mean(A - B))
        if mean_diff < lowest_diff:
            lowest_diff = mean_diff
            minDiffVecs = [A,B]
    
    return minDiffVecs
    
if __name__== "__main__":

    if len(sys.argv) > 1:
        movie_file = sys.argv[1]
    else:
       movie_file = gaitFunctions.select_movie_file()
       
    print('Movie is ' + movie_file)

    main(movie_file)