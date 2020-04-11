# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 04:27:17 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing



df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t01.emg.csv").iloc[0:73,:]




index = pd.date_range('1/1/2000', periods=9, freq='T')

series = pd.Series([32165,32191,32351,32509,32823,32615,33009,33159,33318], index=index)

downscale = series.resample('3T').mean()

upscale = series.resample('30S').pad()

fig, ax = plt.subplots(3)

ax[0].plot(index, series)
ax[0].set_title("normal")

ax[1].plot(upscale.index, upscale)
ax[1].set_title("upscale")

ax[2].plot(downscale.index, downscale)
ax[2].set_title("downscale")

plt.show()
