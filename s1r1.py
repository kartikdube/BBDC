# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:15:39 2020

@author: dubek
"""


import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

df1 = pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\s01t01emg_with_labels.csv')
df = df1.iloc[:,5:10]
df["ts"] = df1.iloc[:,0:1]
