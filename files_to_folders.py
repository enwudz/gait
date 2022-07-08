#!/usr/bin/python
import glob
import os
import shutil

# list (movie) files in a folder
# make a folder for each file and move the file to the folder

wd = os.getcwd()

file_type = 'mov'

file_list = glob.glob('*.' + file_type)

for file in file_list:
    file_path = os.path.join(wd, file)
    file_stem = file.split('.')[0]

    dir_path = os.path.join(wd, file_stem)

    os.mkdir(dir_path)
    shutil.move(file_path, dir_path)
    

