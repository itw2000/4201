__author__ = 'Ian'

from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('taxi_data_sample.csv')
tip_percent = data['tip_amount']/data['total_amount']*100
data = pd.concat([data, tip_percent],axis=1)
data = data.rename(columns = {0:'tip_percent'})

columns = ["passenger_count","trip_distance","rate_code","surcharge","tolls_amount","fare_amount"]
tips = data["tip_percent"].values
design = data[list(columns)].values
design = sm.add_constant(design)

#EDA
data.describe()
plt.plot(data["tolls_amount"], data["tip_percent"])

#Sci-Kit learn lin-reg
mdl = linear_model.LinearRegression()
mdl.fit(design, tips)

#Statsmodels linreg
model = sm.OLS(tips, design)
results = model.fit()
print(results.summary())

#LASSO model
ls_model = linear_model.Lasso(alpha = 0.1)
ls_model.fit(design, tips)
ls_model.coef_