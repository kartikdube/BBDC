# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 05:33:34 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing

df1 = (pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\labels-train.csv', header = None))
df1 = df1.set_axis(['V', 'W', 'X', 'Y'], axis=1, inplace=False)

df_r = df1.iloc[40:113]
df_r = df_r.reset_index(drop=True)

l_df = (np.array((pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\labels-train.csv',squeeze=True, header = None)).iloc[:40,2:3])).flatten()

r_df = (np.array((pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\labels-train.csv',squeeze=True, header = None)).iloc[40:113,2:3])).flatten()

comb_arr = np.array([0,0,"la-nothing_ra-nothing"])
emg_comb_arr = np.array([0,0])

for i in range(0,max(len(l_df),len(r_df))-1):
    
    if i == 0 :
        str1 = df1.Y[np.where(df1.W == 7.37)[0][0]] + '_' + df1.Y[np.where(df1.W == 7.37)[0][1]]
        stk1 = np.array([l_df[0],r_df[0],str1])
        comb_arr = np.vstack((comb_arr,stk1))
        
    elif i == max(len(l_df),len(r_df)) - 1:
        strk_slast = np.array([l_df[-2],r_df[-2],"la-nothing_ra-nothing"])
        comb_arr = np.vstack((comb_arr,stk1))
         
    else:
        closest_value = min(l_df, key= lambda l_value : abs(l_value - r_df[i]))
        str2 = df1.Y[np.where(df1.W == closest_value)[0][0]] + '_' + df_r.Y[np.where(df_r.W == r_df[i])[0][0]]
        stk2 = np.array([closest_value,r_df[i],str2])
        comb_arr = np.vstack((comb_arr,stk2))
        
    
    print(i)










