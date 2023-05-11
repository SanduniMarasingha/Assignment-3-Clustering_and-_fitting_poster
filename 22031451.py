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
from scipy import optimize
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import errors as err
import seaborn as sns

# Read in Climate_Change.csv file and skip first 4 rows
df_Co2Emmision = pd.read_csv("Co2emmision.csv", skiprows=4)
df_Co2Emmision = df_Co2Emmision.iloc[:, [0, 1]].join(
    df_Co2Emmision.iloc[:, 32:62]).reset_index(drop=True)
df_Co2Emmision.describe()

# Read in forestArea.csv file and skip first 4 rows
df_forest = pd.read_csv("forestArea.csv", skiprows=4)
df_forest = df_forest.drop(['Country Code', 'Indicator Code'], axis=1)
df_forest = df_forest.iloc[:, [0, 1]].join(
    df_forest.iloc[:, 32:62]).reset_index(drop=True)
df_forest.describe()

# Merge two dataframes together
combine_df = pd.concat([df_Co2Emmision, df_forest])
combine_df

combine_df.to_excel("new11.xlsx")

# Choose 2017 column of each dataframe
df_Co2Emmision = df_Co2Emmision[df_Co2Emmision["2017"].notna()]
df_forest = df_forest[df_forest["2017"].notna()]

# Choose only essential columns and take a copy
df_Co2Emmision_2017 = df_Co2Emmision[["Country Name", "2017"]].copy()
df_forest_2017 = df_forest[["Country Name", "2017"]].copy()

# Merge Forest Area and Co2 emission columns on Country name
df_forest_co2_2017 = pd.merge(
    df_Co2Emmision_2017, df_forest_2017, on="Country Name", how="outer")

# Rename axis
df_forest_co2_2017 = df_forest_co2_2017.dropna()
df_forest_co2_2017 = df_forest_co2_2017.rename(
    columns={"2017_x": "Forest Area", "2017_y": "CO2 Emmision"})
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
pd.plotting.scatter_matrix(
    df_forest_co2_2017, alpha=0.8, figsize=(10, 10), color="#A52A2A")
plt.tight_layout()
plt.show()


print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    # fit done on x,y pairs
    kmeans.fit(df_forest_co2_2017[['Forest Area', 'CO2 Emmision']])
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(
        df_forest_co2_2017[['Forest Area', 'CO2 Emmision']], labels))

# Drop non-numeric columns
df_forest_co2_2017_numeric = df_forest_co2_2017.select_dtypes(include='number')

# Fit K-means clustering algorithm
kmeans = KMeans(n_clusters=3, random_state=0).fit(df_forest_co2_2017_numeric)

# Add the cluster labels to the data
df_forest_co2_2017['cluster'] = kmeans.labels_

# Select only the numeric columns
df_forest_co2_2017_numeric = df_forest_co2_2017.select_dtypes(include=[
                                                              np.number])

# Normalize the data
scaler = StandardScaler()
df_forest_co2_2017_norm = scaler.fit_transform(df_forest_co2_2017_numeric)

# Convert the normalized data back into a DataFrame
df_forest_co2_2017_norm = pd.DataFrame(
    df_forest_co2_2017_norm, columns=df_forest_co2_2017_numeric.columns)

# Add the non-numeric columns back to the DataFrame
df_forest_co2_2017_norm = pd.concat([df_forest_co2_2017[df_forest_co2_2017.columns.difference(
    df_forest_co2_2017_numeric.columns)], df_forest_co2_2017_norm], axis=1)

cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]

# Visualize the clusters using a scatter plot
plt.scatter(df_forest_co2_2017['CO2 Emmision'], df_forest_co2_2017['Forest Area'],
            c=df_forest_co2_2017['cluster'], cmap='viridis')
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Forest Area")
plt.ylabel("CO2 Emmision")
plt.title("Clustering - Forest Area vs Co2 emission")
plt.show()

# Drop non-numeric columns
df_forest_co2_2017_numeric = df_forest_co2_2017.select_dtypes(include='number')

# Normalize the data
scaler = StandardScaler()
df_forest_co2_2017_norm = scaler.fit_transform(df_forest_co2_2017_numeric)

# Calculate SSE for each K
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_forest_co2_2017_norm)
    sse.append(kmeans.inertia_)

# Plot SSE vs K
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Clustering - Forest Area vs Co2 emission')
plt.show()

# Fitting Forest Area for Sri Lanka
df_forest_trans = df_forest.transpose()
df_forest_trans.columns = df_forest_trans.iloc[0]
df_forest_trans = df_forest_trans.drop(df_forest_trans.index[[0, 1]])
df_forest_Sri = df_forest_trans[['Sri Lanka']]
df_forest_Sri = df_forest_Sri.dropna()
df_forest_Sri = df_forest_Sri.reset_index()
df_forest_Sri = df_forest_Sri.rename(
    columns={'index': 'Year', 'Sri Lanka': 'Forest Area'})
df_forest_Sri['Year'] = df_forest_Sri['Year'].astype(int)
df_forest_Sri['Forest Area'] = df_forest_Sri['Forest Area'].astype(float)

df_forest_Sri

# Exponenential Growth


def exp_growth(t, scale, growth):
    """Calculate exponential growth function value at time t.

    Parameters
    ----------
    t : float or array_like
        The time(s) at which to evaluate the function.
    scale : float
        The initial value of the function at time t=1990.
    growth : float
        The exponential growth rate.

    Returns
    -------
    float or ndarray
        The value(s) of the exponential growth function at time(s) t.
    """
    f = scale * np.exp(growth * (t-1990))
    return f


popt, pcorr = opt.curve_fit(
    exp_growth, df_forest_Sri["Year"], df_forest_Sri["Forest Area"])

print("Fit parameter", popt)

df_forest_Sri["pop_exp"] = exp_growth(df_forest_Sri["Year"], *popt)
plt.figure()
plt.plot(df_forest_Sri["Year"], df_forest_Sri["Forest Area"],
         label="data", color='#458B00')
plt.plot(df_forest_Sri["Year"], df_forest_Sri["pop_exp"],
         label="fit", color='#DC143C')

plt.legend()
plt.title("first fit attempt")
plt.show()


def logistic(t, n0, g, t0):
    """Calculates the logistic function with the given parameters.
    Parameters
    ----------
    t : float or array-like
     The input values for which to calculate the logistic function.
     n0 : float
     The initial population size at time t0.
     g : float
     The growth rate of the population.
     t0 : float
     The time at which the population starts to grow exponentially.

     Returns
     -------
     f : float or array-like
     The output values of the logistic function for the input values t.

     Notes
     -----
     The logistic function is defined as:

     f(t) = n0 / (1 + exp(-g * (t - t0)))

 where:
     - n0 is the initial population size at time t0,
     - g is the growth rate of the population, and
     - t0 is the time at which the population starts to grow exponentially.
 """
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


# Extract the year and total Forest Area as numpy arrays
x = df_forest_Sri['Year']
y = df_forest_Sri['Forest Area'].values

# Define the initial guess for the parameters (n0, g, t0)
n0 = np.mean(y)
g = 0.01
t0 = np.median(x)
p0 = [n0, g, t0]

# Fit the logistic model to the data using the Levenberg-Marquardt algorithm
params, covar = opt.curve_fit(logistic, x, y, p0, method='lm', maxfev=10000)

# Print the optimal parameters and their estimated standard deviation
print("Optimal parameters: ", params)
print("Standard deviation: ", np.sqrt(np.diag(covar)))

# find a feasible start value the pedestrian way
# initial guess
popt = [max(y), 1, np.median(x)]
df_forest_Sri["pop_exp"] = logistic(df_forest_Sri["Year"], *popt)
plt.figure()
plt.plot(df_forest_Sri["Year"], df_forest_Sri["Forest Area"], label="data")
plt.plot(df_forest_Sri["Year"], df_forest_Sri["pop_exp"], label="fit")
plt.legend()
plt.title("Logistic Model Fit to Forest Area Data in Sri Lanka with Initial Guess")
plt.show()

popt, pcorr = opt.curve_fit(logistic, x, y, p0, maxfev=10000)
print("Fit parameter", popt)

df_forest_Sri["pop_logistics"] = logistic(df_forest_Sri["Year"], *popt)

plt.figure()
plt.title("logistics function")
plt.plot(df_forest_Sri["Year"], df_forest_Sri["Forest Area"], label="data")
plt.plot(df_forest_Sri["Year"], df_forest_Sri["pop_logistics"], label="fit")
plt.legend()
plt.show()

print("Population in")
print("2030:", logistic(2030, *popt) / 1.0e6, "Mill.")
print("2040:", logistic(2040, *popt) / 1.0e6, "Mill.")
print("2050:", logistic(2050, *popt) / 1.0e6, "Mill.")

# Define the logistic function


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


# Generate some test data
t = np.linspace(0, 10, 100)
y_true = logistic(t, 50, 1, 5) + np.random.normal(scale=5, size=t.size)

# Fit the logistic model to the data
p0 = [max(y_true), 1, np.median(t)]
params, covar = optimize.curve_fit(logistic, t, y_true, p0)

# Plot the results
plt.plot(t, y_true, 'ko', label='True data')
plt.plot(t, logistic(t, *params), 'r-', label='Fit')
plt.legend()
plt.show()

# Fit the logistic model to the data
p0 = [max(df_forest_Sri["Forest Area"]), 1,
      np.median(df_forest_Sri["Forest Area"])]
params, covar = optimize.curve_fit(
    logistic, df_forest_Sri["Forest Area"], df_forest_Sri["pop_logistics"], p0)

# Calculate the predicted y values for the fit
y_fit = logistic(df_forest_Sri["Forest Area"], *params)

# Fit the logistic model to the data
p0 = [max(y_true), 1, np.median(t)]
params, covar = optimize.curve_fit(logistic, t, y_true, p0)

# Plot the results
plt.plot(t, y_true, 'ko', label='True data')
plt.plot(t, logistic(t, *params), 'r-', label='Fit')
plt.legend()
plt.show()

# Define the logistic function


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))

    return f


# Fit the logistic model to the data
p0 = [max(df_forest_Sri["Forest Area"]), 1,
      np.median(df_forest_Sri["Forest Area"])]
params, covar = optimize.curve_fit(
    logistic, df_forest_Sri["Forest Area"], df_forest_Sri["pop_logistics"], p0)

# Fit the logistic model to the data
p0 = [max(y_true), 1, np.median(t)]
params, covar = optimize.curve_fit(logistic, t, y_true, p0)

# Plot the results
plt.plot(t, y_true, 'k-', label='True data')
plt.plot(t, logistic(t, *params), 'r-', label='Fit')

plt.legend()
plt.show()

# Select one country from each cluster
clustered_countries = df_forest_co2_2017.groupby(
    'cluster').apply(lambda x: x.sample(n=1))

# Print the selected countries
print(clustered_countries[['Country Name', 'cluster']])

# Compare the features of the selected countries
print(clustered_countries[['Country Name', 'Forest Area', 'CO2 Emmision']])

# Select countries from different clusters
cluster0_countries = df_forest_co2_2017[df_forest_co2_2017['cluster'] == 0].sample(
    n=3)
cluster1_countries = df_forest_co2_2017[df_forest_co2_2017['cluster'] == 1].sample(
    n=3)

# Compare the features of the selected countries
print(cluster0_countries[['Country Name', 'Forest Area', 'CO2 Emmision']])
print(cluster1_countries[['Country Name', 'Forest Area', 'CO2 Emmision']])

# Plot the features of the countries in each cluster
sns.pairplot(clustered_countries, hue='cluster',
             vars=['Forest Area', 'CO2 Emmision'])
