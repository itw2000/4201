__author__ = 'Ian'

from sklearn import linear_model
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import statsmodels.api as sms
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

data = pd.read_csv('taxi_data_sample_withoutoutliers.csv')
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data = data.rename(columns = {'1/2/3/4':'tod'})
data = data.rename(columns = {'0/1(offpeak/peak)':'peak'})
data = data.rename(columns = {'tip percent':'tip_percent'})



columns = ["passenger_count","trip_distance","surcharge","tolls_amount","fare_amount","peak","tod","rate_code"]
tips = data["tip percent"].values

design = data[list(columns)].values
design = sms.add_constant(design)

#EDA
data.describe()

#Statsmodels linreg
formula = 'tip_percent ~ passenger_count + trip_distance + surcharge + tolls_amount + fare_amount + C(rate_code) + peak + C(tod)'
model = smf.ols(formula=formula, data=data)
results = model.fit()
print results.summary()

print('Parameters: ', results.params)
print('Standard errors: ', results.bse)
print('Predicted values: ', results.predict())

preds = results.predict()
residuals = results.resid
np.savetxt('residuals.csv', residuals, delimiter=",")
np.savetxt('fitted_values.csv', preds, delimiter=",")
coef = results.params
se = results.bse

np.savetxt('coef.csv', coef, delimiter=",")
np.savetxt('se.csv', se, delimiter=",")

print summary_col([results])




#LASSO model
ls_model = linear_model.Lasso(alpha = 0.1)
ls_model.fit(design, tips)
coef = list(ls_model.coef_)
print ', '.join(map(str,coef))


#Reduced model
formula2 = 'tip_percent ~ passenger_count + trip_distance + surcharge + tolls_amount + fare_amount + C(tod)'
model = smf.ols(formula=formula2, data=data)
results2 = model.fit()
print results2.summary()

preds = results2.predict()
residuals = results2.resid
coef = results2.coef()

np.savetxt('reduced_residuals.csv', residuals, delimiter=",")
np.savetxt('reduced_fitted_values.csv', preds, delimiter=",")
np.savetxt('coef.csv', preds, delimiter=",")


print summary_col([results, results2],stars=True,float_format='%0.2f',model_names=['Full','Reduced']).as_latex()
