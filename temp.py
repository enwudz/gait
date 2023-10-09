# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import os
# import shutil
# import glob
# import pandas as pd
import gaitFunctions
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.path as mpath
# import matplotlib.patches as mpatches


def main():
    
    # flist = gaitFunctions.getFileList()
    sampmov = 'ea_14jul_tardigrade42_day2.mp4'
    file_stem = sampmov.split('.')[0]
    frame_folder = file_stem + '_frames'
    gaitFunctions.saveFrames(frame_folder,sampmov)
    

main()    