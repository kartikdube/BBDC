# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 07:24:15 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing

ending_df = pd.read_pickle("ending_DF.pkl")
ending_df.iloc[:,0:1] = (ending_df.iloc[:,0:1]).astype(float)

s06t01_df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\test\\emg\\s06t01.emg.csv")
s06t02_df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\test\\emg\\s06t02.emg.csv")
s06t03_df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\test\\emg\\s06t03.emg.csv")
s06t04_df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\test\\emg\\s06t04.emg.csv")
s06t05_df = pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\test\\emg\\s06t05.emg.csv")

t_list = [s06t01_df, s06t02_df, s06t03_df, s06t04_df, s06t05_df]
t_name_list = ['s06t01_df', 's06t02_df', 's06t03_df', 's06t04_df', 's06t05_df']

n = 1200

result_list = np.array(['recName',0,0,'label'])

for t_, name_ in zip(t_list, t_name_list) :
    
    for i,z in zip(range(0,int(len(t_)/n -1)),range(0,int(len(t_)/n - 2),2) ) :
        
        compare_result = ''
        
        if i == 0:
            small_df = t_.iloc[0:1200,1:]
        else:
            small_df = t_.iloc[i*n:(i+1)*n,1:]
        
        std_sum = (sum(small_df.std()) ) / 8
        
        
        
        if std_sum < 400:
            compare_result = 'la-nothing_ra-nothing'
        
        else:
            closest_value = min((np.array(ending_df.iloc[:,0:1])).flatten(), key= lambda r_value : abs(r_value - std_sum))
            compare_result = (np.array(ending_df.loc[ending_df['Average Standard Deviation'] == closest_value]))[0][1]
            
        
        print(i)
    
        stk = np.array([name_,z,z+2,compare_result])
        result_list = np.vstack((result_list,stk))
            
result_DF = (pd.DataFrame(result_list,columns = ['rec_name','start (in seconds)','end (in seconds)','label'])).iloc[1:,:]

result_DF.to_csv('Result.csv')
            
            
            
            
            
            
            
            
            
            
            
