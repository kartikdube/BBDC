# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:00:18 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

df1 = pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t01.emg.csv')

X = np.array(df1.iloc[:,6:7])

model = "l2"  
algo = rpt.Window(width=1500, model=model).fit(X)
my_bkps = algo.predict(n_bkps=40)
rpt.show.display(X, my_bkps, figsize=(25, 20))
plt.title('Change Point Detection: Window-Based Search Method')
plt.show()