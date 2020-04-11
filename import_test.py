# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:23:41 2020

@author: dubek
"""


import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t01.emg.csv")