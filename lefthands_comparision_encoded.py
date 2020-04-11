# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:34:24 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing

df1 = (pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\s1r1_left.csv')).iloc[:70489,1:]

labelencoder = preprocessing.LabelEncoder()
df1['left_hand_labels'] = labelencoder.fit_transform(df1['left_hand_labels'])

fig, ax = plt.subplots(4)

ax[0].plot(df1["ts"], df1["fa-o-t-l"])
ax[0].set_title("outside top of left forearm")

ax[1].plot(df1["ts"], df1["fa-i-t-l"])
ax[1].set_title("top of inside of left forearm")

ax[2].plot(df1["ts"], df1["fa-i-b-l"])
ax[2].set_title("inner bottom of left forearm")

ax[3].plot(df1["ts"], df1["fa-o-b-l"])
ax[3].set_title("outside bottom of left forearm")


enc = df1["left_hand_labels"]
p_n = enc[0]
x_n = 0

for en in enc:
    if en != p_n:
        ax[0].axvline(x = df1["ts"][x_n], color = "r")
        ax[1].axvline(x = df1["ts"][x_n], color = "r")
        ax[2].axvline(x = df1["ts"][x_n], color = "r")
        ax[3].axvline(x = df1["ts"][x_n], color = "r")
        p_n = en
    x_n += 1
plt.show()