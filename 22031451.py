# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:31:57 2023

@author: Sanduni Sakunthala
"""

import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as skmet

import matplotlib.pyplot as plt
import cluster_tools as ct

from scipy.optimize import curve_fit
import scipy.optimize as opt
from sklearn.cluster import KMeans

# Read in Climate_Change.csv file and skip first 4 rows
df_Co2Emmision = pd.read_csv("Co2emmision.csv", skiprows=4)
df_Co2Emmision=df_Co2Emmision.iloc[:,[0,1]].join(df_Co2Emmision.iloc[:, 32:62]).reset_index(drop=True)
df_Co2Emmision.describe()

df_forest = pd.read_csv("forestArea.csv",skiprows=4)
df_forest = df_forest.drop(['Country Code', 'Indicator Code'], axis=1)
df_forest=df_forest.iloc[:,[0,1]].join(df_forest.iloc[:, 32:62]).reset_index(drop=True)
df_forest.describe()

combine_df = pd.concat([df_Co2Emmision, df_forest])
combine_df

combine_df.to_excel("new11.xlsx")

#Choose 2019 column of each dataframe
df_Co2Emmision = df_Co2Emmision[df_Co2Emmision["2017"].notna()]
df_forest = df_forest[df_forest["2017"].notna()]

#Choose only essential columns and take a copy
df_Co2Emmision_2017 =df_Co2Emmision[["Country Name", "2017"]].copy()
df_forest_2017 = df_forest[["Country Name", "2017"]].copy()

#Merge renewable energy and Co2 emission columns on Country name
df_forest_co2_2017= pd.merge(df_Co2Emmision_2017, df_forest_2017, on="Country Name", how="outer")
df_forest_co2_2017.to_excel("df_forest_co2_2017_co2.xlsx")

#Rename axis
df_forest_co2_2017 = df_forest_co2_2017.dropna() 
df_forest_co2_2017 = df_forest_co2_2017.rename(columns={"2017_x":"Forest Area", "2017_y":"CO2 Emmision"})
df_forest_co2_2017.to_excel("df_XY_VALUES.xlsx")

# Compute the correlation matrix
corr_matrix = df_forest_co2_2017.corr()
corr_matrix

# Plot the correlation matrix as a heatmap
plt.imshow(corr_matrix, cmap='YlGnBu', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.show()

# Plot the scatterplot matrix
pd.plotting.scatter_matrix(df_forest_co2_2017, alpha=0.8, figsize=(10, 10),color= "#A52A2A")
plt.tight_layout()
plt.show()

# Drop non-numeric columns
df_forest_co2_2017_numeric = df_forest_co2_2017.select_dtypes(include='number')

# Fit K-means clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_forest_co2_2017_numeric)

# Add the cluster labels to the data
df_forest_co2_2017['cluster'] = kmeans.labels_

# Visualize the clusters using a scatter plot
plt.scatter(df_forest_co2_2017['CO2 Emmision'], df_forest_co2_2017['Forest Area'], c=df_forest_co2_2017['cluster'], cmap='viridis')
plt.show()