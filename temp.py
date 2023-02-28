# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import gaitFunctions
import pandas as pd
import matplotlib.pyplot as plt

f,a = plt.subplots(nrows=1, ncols=1, figsize=(3,3))

a = gaitFunctions.gaitStyleLegend(a, 'lateral')
plt.show()
