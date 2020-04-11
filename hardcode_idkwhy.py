# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 05:11:48 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing


df = (pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\labels-train.csv', header = None)).set_axis(['V', 'W', 'X', 'Y'], axis=1, inplace=False)

s1t1_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t01.emg.csv", header = None, skiprows = 1)).set_index(0)
s1t2_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t02.emg.csv", header = None, skiprows = 1)).set_index(0)
s1t3_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t03.emg.csv", header = None, skiprows = 1)).set_index(0)
s1t4_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t04.emg.csv", header = None, skiprows = 1)).set_index(0)
s1t5_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t05.emg.csv", header = None, skiprows = 1)).set_index(0)
s1t6_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s01t06.emg.csv", header = None, skiprows = 1)).set_index(0)

s2t1_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t01.emg.csv", header = None, skiprows = 1)).set_index(0)
s2t2_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t02.emg.csv", header = None, skiprows = 1)).set_index(0)
s2t3_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t03.emg.csv", header = None, skiprows = 1)).set_index(0)
s2t4_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t04.emg.csv", header = None, skiprows = 1)).set_index(0)
s2t5_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t05.emg.csv", header = None, skiprows = 1)).set_index(0)
s2t6_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s02t06.emg.csv", header = None, skiprows = 1)).set_index(0)

s3t1_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s03t01.emg.csv", header = None, skiprows = 1)).set_index(0)
s3t2_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s03t02.emg.csv", header = None, skiprows = 1)).set_index(0)
s3t3_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s03t03.emg.csv", header = None, skiprows = 1)).set_index(0)
s3t4_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s03t04.emg.csv", header = None, skiprows = 1)).set_index(0)
s3t5_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s03t05.emg.csv", header = None, skiprows = 1)).set_index(0)

s4t1_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s04t01.emg.csv", header = None, skiprows = 1)).set_index(0)
s4t2_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s05t02.emg.csv", header = None, skiprows = 1)).set_index(0)

s5t1_df = (pd.read_csv("K:\\AA JU DE\\BBDC\\bbdc_2020\\train\\emg\\s05t06.emg.csv", header = None, skiprows = 1)).set_index(0)

s1t1_l , s1t1_r, s1t1_df_l, s1t1_df_r = np.array((df.iloc[:40,2:3])).flatten() , np.array((df.iloc[40:113,2:3])).flatten() , (df.iloc[:40]).reset_index(drop=True), (df.iloc[40:113]).reset_index(drop=True)
s1t2_l , s1t2_r, s1t2_df_l, s1t2_df_r = np.array((df.iloc[113:162,2:3])).flatten() , np.array((df.iloc[162:279,2:3])).flatten() , (df.iloc[113:162]).reset_index(drop=True), (df.iloc[162:279]).reset_index(drop=True)
s1t3_l , s1t3_r, s1t3_df_l, s1t3_df_r = np.array((df.iloc[279:308,2:3])).flatten() , np.array((df.iloc[308:393,2:3])).flatten() , (df.iloc[279:308]).reset_index(drop=True), (df.iloc[308:393]).reset_index(drop=True)
s1t4_l , s1t4_r, s1t4_df_l, s1t4_df_r = np.array((df.iloc[393:422,2:3])).flatten() , np.array((df.iloc[422:495,2:3])).flatten() , (df.iloc[393:422]).reset_index(drop=True), (df.iloc[422:495]).reset_index(drop=True)
s1t5_l , s1t5_r, s1t5_df_l, s1t5_df_r = np.array((df.iloc[495:546,2:3])).flatten() , np.array((df.iloc[546:621,2:3])).flatten() , (df.iloc[495:546]).reset_index(drop=True), (df.iloc[546:621]).reset_index(drop=True)
s1t6_l , s1t6_r, s1t6_df_l, s1t6_df_r = np.array((df.iloc[621:650,2:3])).flatten() , np.array((df.iloc[650:693,2:3])).flatten() , (df.iloc[621:650]).reset_index(drop=True), (df.iloc[650:693]).reset_index(drop=True)

s2t1_l , s2t1_r, s2t1_df_l, s2t1_df_r = np.array((df.iloc[693:752,2:3])).flatten() , np.array((df.iloc[752:832,2:3])).flatten() , (df.iloc[693:752]).reset_index(drop=True), (df.iloc[752:832]).reset_index(drop=True)
s2t2_l , s2t2_r, s2t2_df_l, s2t2_df_r = np.array((df.iloc[832:923,2:3])).flatten() , np.array((df.iloc[923:987,2:3])).flatten() , (df.iloc[832:923]).reset_index(drop=True), (df.iloc[923:987]).reset_index(drop=True)
s2t3_l , s2t3_r, s2t3_df_l, s2t3_df_r = np.array((df.iloc[987:1048,2:3])).flatten() , np.array((df.iloc[1048:1134,2:3])).flatten() , (df.iloc[987:1048]).reset_index(drop=True), (df.iloc[1048:1134]).reset_index(drop=True)
s2t4_l , s2t4_r, s2t4_df_l, s2t4_df_r = np.array((df.iloc[1134:1163,2:3])).flatten() , np.array((df.iloc[1163:1205,2:3])).flatten() , (df.iloc[1134:1163]).reset_index(drop=True), (df.iloc[1163:1205]).reset_index(drop=True)
s2t5_l , s2t5_r, s2t5_df_l, s2t5_df_r = np.array((df.iloc[1205:1287,2:3])).flatten() , np.array((df.iloc[1287:1349,2:3])).flatten() , (df.iloc[1205:1287]).reset_index(drop=True), (df.iloc[1287:1349]).reset_index(drop=True)
s2t6_l , s2t6_r, s2t6_df_l, s2t6_df_r = np.array((df.iloc[1349:1401,2:3])).flatten(), np.array((df.iloc[1401:1470,2:3])).flatten(), (df.iloc[1349:1401]).reset_index(drop=True), (df.iloc[1401:1470]).reset_index(drop=True)

s3t1_l , s3t1_r, s3t1_df_l, s3t1_df_r = np.array((df.iloc[1470:1515,2:3])).flatten() , np.array((df.iloc[1515:1590,2:3])).flatten() , (df.iloc[1470:1515]).reset_index(drop=True), (df.iloc[1515:1590]).reset_index(drop=True)
s3t2_l , s3t2_r, s3t2_df_l, s3t2_df_r = np.array((df.iloc[1590:1635,2:3])).flatten() , np.array((df.iloc[1635:1694,2:3])).flatten() , (df.iloc[1590:1635]).reset_index(drop=True), (df.iloc[1635:1694]).reset_index(drop=True)
s3t3_l , s3t3_r, s3t3_df_l, s3t3_df_r = np.array((df.iloc[1694:1750,2:3])).flatten() , np.array((df.iloc[1750:1842,2:3])).flatten() , (df.iloc[1694:1750]).reset_index(drop=True), (df.iloc[1750:1842]).reset_index(drop=True)
s3t4_l , s3t4_r, s3t4_df_l, s3t4_df_r = np.array((df.iloc[1843:1884,2:3])).flatten() , np.array((df.iloc[1884:1941,2:3])).flatten() , (df.iloc[1843:1884]).reset_index(drop=True), (df.iloc[1884:1941]).reset_index(drop=True)
s3t5_l , s3t5_r, s3t5_df_l, s3t5_df_r = np.array((df.iloc[1941:1993,2:3])).flatten() , np.array((df.iloc[1993:2066,2:3])).flatten() , (df.iloc[1941:1993]).reset_index(drop=True), (df.iloc[1993:2066]).reset_index(drop=True)

s4t1_l , s4t1_r, s4t1_df_l, s4t1_df_r = np.array((df.iloc[2066:2105,2:3])).flatten() , np.array((df.iloc[2105:2160,2:3])).flatten() , (df.iloc[2066:2105]).reset_index(drop=True), (df.iloc[2105:2160]).reset_index(drop=True)
s4t2_l , s4t2_r, s4t2_df_l, s4t2_df_r = np.array((df.iloc[2160:2234,2:3])).flatten() , np.array((df.iloc[2234:2380,2:3])).flatten() , (df.iloc[2160:2234]).reset_index(drop=True), (df.iloc[2234:2380]).reset_index(drop=True)

s5t1_l , s5t1_r, s5t1_df_l, s5t1_df_r = np.array((df.iloc[2380:2412,2:3])).flatten() , np.array((df.iloc[2412:2488,2:3])).flatten() , (df.iloc[2380:2412]).reset_index(drop=True), (df.iloc[2412:2488]).reset_index(drop=True)


l_list = [s1t1_l, s1t2_l, s1t3_l, s1t4_l, s1t5_l, s1t6_l,  s2t1_l, s2t2_l , s2t3_l , s2t4_l, s2t5_l, s2t6_l , s3t1_l , s3t2_l , s3t3_l , 
          s3t4_l, s3t5_l , s4t1_l , s4t2_l , s5t1_l]
r_list = [s1t1_r , s1t2_r , s1t3_r , s1t4_r , s1t5_r , s1t6_r , s2t1_r , s2t2_r , s2t3_r , s2t4_r , s2t5_r , s2t6_r , s3t1_r , s3t2_r,
          s3t3_r , s3t4_r , s3t5_r , s4t1_r , s4t2_r , s5t1_r]
df_l_list = [s1t1_df_l, s1t2_df_l, s1t3_df_l, s1t4_df_l, s1t5_df_l, s1t6_df_l,  s2t1_df_l, s2t2_df_l , s2t3_df_l , s2t4_df_l, s2t5_df_l, s2t6_df_l , s3t1_df_l , s3t2_df_l , s3t3_df_l , 
          s3t4_df_l, s3t5_df_l , s4t1_df_l , s4t2_df_l , s5t1_df_l]
df_r_list = [s1t1_df_r , s1t2_df_r , s1t3_df_r , s1t4_df_r , s1t5_df_r , s1t6_df_r , s2t1_df_r , s2t2_df_r , s2t3_df_r , s2t4_df_r , s2t5_df_r , s2t6_df_r , s3t1_df_r , s3t2_df_r,
          s3t3_df_r , s3t4_df_r , s3t5_df_r , s4t1_df_r , s4t2_df_r , s5t1_df_r]

df_list = [s1t1_df, s1t2_df, s1t3_df,s1t4_df,s1t5_df,s1t6_df,s2t1_df,s2t2_df,s2t3_df,s2t4_df,s2t5_df,s2t6_df, s3t1_df, s3t2_df,s3t3_df,
           s3t4_df,s3t4_df,s4t1_df, s4t2_df, s5t1_df]

name_list = ['s1t1_ts', 's1t2_ts', 's1t3_ts', 's1t4_ts', 's1t5_ts', 's1t6_ts',  's2t1_ts', 's2t2_ts' , 's2t3_ts' , 's2t4_ts', 's2t5_ts', 's2t6_ts' , 's3t1_ts' , 's3t2_ts' , 's3t3_ts' , 
          's3t4_ts', 's3t5_ts' , 's4t1_ts' , 's4t2_ts' , 's5t1_ts']

comb_arr = np.array([0,0,"labels"])
master_df = np.array([[0],[0],'labels'])
std_df = np.array([0,0,"labels"])
flag = 0

for l_, r_ ,dfl_ , dfr_, n_l, df_  in zip(l_list, r_list, df_l_list, df_r_list,name_list, df_list):
    
    str0 = "la-nothing_ra-nothing"
    stk0 = np.array([0,0,str0])
    comb_arr = np.vstack((comb_arr,stk0))
    
    if len(l_) > len(r_):
        flag = 0
    else:
        flag = 1
    
    for i in range(0,max(len(l_),len(r_)) -1):
    
        if i == 0 :
            str1 = dfl_.Y[2] + '_' + dfr_.Y[2]
            stk1 = np.array([l_[0],r_[0],str1])
            comb_arr = np.vstack((comb_arr,stk1))
        
        elif i == max(len(l_),len(r_)) - 1:
            strk_slast = np.array([l_[-2],r_[-2],"la-nothing_ra-nothing"])
            comb_arr = np.vstack((comb_arr,stk1))
         
        else:
            
            if flag == 1:
                closest_value = min(l_, key= lambda l_value : abs(l_value - r_[i]))
                str2 = dfl_.Y[np.where(dfl_.X == closest_value)[0][0]] + '_' + dfr_.Y[i]
                stk2 = np.array([closest_value,r_[i],str2])
                comb_arr = np.vstack((comb_arr,stk2))
                
                if float(comb_arr[-2][0])  != float(comb_arr[-1][0]):
                    x_ = (df_.loc[float(comb_arr[-2][0]):float(comb_arr[-1][0])]).iloc[:,4:]
                    y_ = (df_.loc[float(comb_arr[-2][1]):float(comb_arr[-1][1])]).iloc[:,:4]
                    
                else:
                    x_ = x_
                    y_ = (df_.loc[float(comb_arr[-2][1]):float(comb_arr[-1][1])]).iloc[:,:4]
                
                std_stk1 = np.array([sum(x_.std()) / 4 , sum(y_.std()) / 4, str2])
                std_df = np.vstack((std_df,std_stk1)) 
                
                z_ = np.array([x_,y_,str2])
                master_df = np.vstack((master_df,z_))
            
                
            else:
                closest_value = min(r_, key= lambda r_value : abs(r_value - l_[i]))
                str2 = dfl_.Y[i] + '_' + dfr_.Y[np.where(dfr_.X == closest_value)[0][0]]
                stk2 = np.array([l_[i],closest_value,str2])
                comb_arr = np.vstack((comb_arr,stk2))
                
                if float(comb_arr[-2][1])  != float(comb_arr[-1][1]):
                    x_ = (df_.loc[float(comb_arr[-2][0]):float(comb_arr[-1][0])]).iloc[:,4:]
                    y_ = (df_.loc[float(comb_arr[-2][1]):float(comb_arr[-1][1])]).iloc[:,:4]
                    
                else:
                    x_ = (df_.loc[float(comb_arr[-2][0]):float(comb_arr[-1][0])]).iloc[:,4:]
                    y_ = y_
                
                z_ = np.array([x_,y_,str2])
                
                
                std_stk1 = np.array([sum(x_.std()) / 4 , sum(y_.std()) / 4, str2])
                std_df = np.vstack((std_df,std_stk1))                
                
                master_df = np.vstack((master_df,z_))
                
                
            
    
    #combDF = pd.DataFrame(comb_arr)
    #combDF.to_pickle('' + n_l + '.pkl')
std_DF = pd.DataFrame(std_df, columns = ['left','right','label'])
encoded_sd_df = std_DF.iloc[1:,:]
le = preprocessing.LabelEncoder()
encoded_sd_df['label'] = le.fit_transform(encoded_sd_df['label'])

encoded_sd_df = encoded_sd_df.reset_index(drop=1)

#averaged
ending_df = np.array([0,'label'])

for i in range(0,36):
    
    filter1 = encoded_sd_df.loc[encoded_sd_df['label'] == i]
    filter1.iloc[:,:2] = filter1.iloc[:,:2].astype(float)
    l_total = (filter1['left'].sum() ) / len(filter1.left)
    r_total = (filter1['right'].sum() ) / len(filter1.right)
    x_total = (r_total + l_total) / 2
    total_ = np.array([x_total,i])
    ending_df = np.vstack((ending_df,total_))
    
ending_DF = (pd.DataFrame(ending_df,columns = ['Average Standard Deviation','label'])).iloc[1:,:]
ending_DF['label'] = (ending_DF['label'].astype(float)).astype(int)
ending_DF.label = le.inverse_transform(ending_DF.label)

ending_DF.to_pickle('ending_DF.pkl')         

combDF = pd.DataFrame(comb_arr)
combDF.to_pickle('comb_DF_TS.pkl')

master_DF = pd.DataFrame(master_df)
master_DF.to_pickle('master_DF.pkl')

std_DF = pd.DataFrame(std_df)
std_DF.to_pickle('std_DF.pkl')



