# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 01:23:44 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing



df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t01.emg.csv", header = None, skiprows = 1)).set_index(0)

df1 = df.iloc[0:1234,:]

downscale = df1.resample('3T').mean()

