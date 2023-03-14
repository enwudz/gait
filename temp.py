# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# import os
# import shutil
# import glob
# import pandas as pd
# import gaitFunctions
import numpy as np

gait_cycle = 1 # in seconds
duty_factor = 0.5 # in fraction of gait cycle
up_frame = duty_factor * gait_cycle
fps = 30

# what is the frame time for when the foot swings?ArithmeticError

num_frames = gait_cycle * fps

frames = np.linspace(1/fps, gait_cycle, num_frames)

uptimes = []
downtimes = [0]

leg_states = ['down' if x <= up_frame else 'up' for x in frames]
print(leg_states)