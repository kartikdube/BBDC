# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 03:32:53 2020

@author: dubek
"""

import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from sklearn import preprocessing

df1 = (pd.read_csv('K:\\AA JU DE\\BBDC\\bbdc_2020\\one\\s1r1_right.csv')).iloc[:70489,1:]

labelencoder = preprocessing.LabelEncoder()
df1['labels'] = labelencoder.fit_transform(df1['labels'])

fig, ax = plt.subplots(4)

ax[0].plot(df1["ts"], df1["fa-o-t-r"])
ax[0].set_title("outside top of right forearm")

ax[1].plot(df1["ts"], df1["fa-i-t-r"])
ax[1].set_title("top of inside of right forearm")

ax[2].plot(df1["ts"], df1["fa-i-b-r"])
ax[2].set_title("inner bottom of right forearm")

ax[3].plot(df1["ts"], df1["fa-o-b-r"])
ax[3].set_title("outside bottom of right forearm")


enc = df1["labels"]
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