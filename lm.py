__author__ = 'Ian'

from sklearn import linear_model
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

data = pd.read_csv('taxi_data_sample.csv')
tip_percent = data['tip_amount']/data['fare_amount']*100
data = pd.concat([data, tip_percent],axis=1)
data = data.rename(columns = {0:'tip_percent'})
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

monthlyMeans = []
months = range(1,13)

for i in range(1, 13):
    mn = np.mean(data[data["pickup_datetime"].dt.month==i]["tip_percent"])
    monthlyMeans.append(mn)

monthlyMeans = np.array(monthlyMeans)
months = np.array(months)
columns = ["passenger_count","trip_distance","surcharge","tolls_amount","fare_amount"]
tips = data["tip_percent"].values

design = data[list(columns)].values
design = sm.add_constant(design)

#EDA
data.describe()

#Statsmodels linreg
formula = 'tip_percent ~ passenger_count + trip_distance + surcharge + tolls_amount + fare_amount + C(rate_code)'
model = smf.ols(formula=formula, data=data)
results = model.fit()
print results.summary()
dfbeta = results.get_influence()


#LASSO model
ls_model = linear_model.Lasso(alpha = 0.1)
ls_model.fit(design, tips)
coef = list(ls_model.coef_)
print ', '.join(map(str,coef))

reducedModel = sm.OLS(tips, design)
reducedResults = reducedModel.fit()
print reducedResults.summary()


#Only trip_dsitance
tripDist = data["trip_distance"].values
distModel = sm.OLS(tips, tripDist)
distResults = distModel.fit()
print distResults.summary()

rush_hour = pd.DataFrame(([0] * 6000), columns=["rush_hour"])
data = pd.concat([data, rush_hour], axis=1)

data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

data.loc[(data["pickup_datetime"].dt.hour >= 7) & (data["pickup_datetime"].dt.hour<=10), 'rush_hour'] = 1
data.loc[(data["pickup_datetime"].dt.hour >= 16) & (data["pickup_datetime"].dt.hour<= 19), 'rush_hour'] = 1

data.loc[data.rush_hour!=1, 'rush_hour'] = 0

#Statsmodels linreg with rushhour dummy
formula = 'tip_percent ~ passenger_count + trip_distance + surcharge + tolls_amount + fare_amount + C(rate_code) + rush_hour'
model2 = smf.ols(formula=formula, data=data)
results = model2.fit()
print results.summary()

#Lin reg with night dummy
nighttime = pd.DataFrame(([0] * 6000), columns=["nighttime"])
data = pd.concat([data, nighttime], axis=1)


data.loc[(data["pickup_datetime"].dt.hour > 19) | (data["pickup_datetime"].dt.hour <= 4), 'nighttime'] = 1

data.loc[data.nighttime!=1, 'nighttime'] = 0

formula = 'tip_percent ~ passenger_count + trip_distance + surcharge + tolls_amount + fare_amount + C(rate_code) + nighttime'
model3 = smf.ols(formula=formula, data=data)
results = model3.fit()
print results.summary()

