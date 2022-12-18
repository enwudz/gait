#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:32:38 2022

@author: iwoods
"""

import glob
import trackCritter
import analyzePath

# movieFiles = glob.glob('*.mov')
# for file in sorted(movieFiles):
#     trackCritter.main(file)

centroidFiles = glob.glob('*tracked*')
for file in sorted(centroidFiles):
    analyzePath.main(file)
    
