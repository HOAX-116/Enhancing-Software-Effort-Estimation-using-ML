#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:07:24 2024

@author: cheera
"""
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/home/cheera/Documents/NITW /desharnais.csv",index_col=0)
corr_matrix=data.corr()
plt.figure(figsize=(11,9))
cmap='coolwarm'
plt.imshow(corr_matrix,cmap=cmap)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        plt.text(i, j, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Pearson Correlation Matrix")
plt.xlabel("Features")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
