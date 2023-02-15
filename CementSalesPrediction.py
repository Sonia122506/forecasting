#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Sales Prediction
import pandas as pd
sales=pd.read_excel("/Users/neerajkumar/Desktop/SONIA/CementData.xlsx")
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
#EDA
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(sales["Sales(in crores)"])
sns.boxplot(sales["Sales(in crores)"])
#pip install feature_engine
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Sales(in crores)"])
winsor.fit_transform(sales[["Sales(in crores)"]])
sales["Sales(in crores)"].plot()
plt.hist(sales["Sales(in crores)"])
sns.boxplot(sales["Sales(in crores)"])
#Stationary Check
import statsmodels.api as sm
decomposition=sm.tsa.seasonal_decompose(x=sales["Sales(in crores)"],model="additive",extrapolate_trend="freq",period=2)
fig=decomposition.plot()
plt.show()

from statsmodels.tsa.stattools import adfuller
adftest=adfuller(sales["Sales(in crores)"])
print("p value of adfuller test:",adftest[1])

#it is stationary


train=sales.head(61)
test=sales.tail(12)
tsa_plots.plot_pacf(sales["Sales(in crores)"],lags=12)
tsa_plots.plot_acf(sales["Sales(in crores)"],lags=12)

#ARIMA with AR=1,MA=4
model=ARIMA(train["Sales(in crores)"],order=(12,0,9))
res=model.fit()

#Forecast for next 12 months
start_index=len(train)
end_index=start_index + 11
forecast_test=res.predict(start=start_index,end=end_index)
print(forecast_test)
import numpy as np
#Evaluate Forecasts
rmse_test=np.sqrt(mean_squared_error(test["Sales(in crores)"],forecast_test))
print("Test RMSE:%.2f"%rmse_test)
import matplotlib.pyplot as plt
train["Sales(in crores)"].plot(legend=True,label="Train",figsize=(10,6))
test["Sales(in crores)"].plot (legend=True,label="Test")
forecast_test.plot(legend=True,label="predictionARIMA")         
plt.plot(test["Sales(in crores)"])
plt.plot(forecast_test,color="red")
plt.show()
#AUTOARIMA
#pip install pmdarima
import pmdarima as pm
ar_model=pm.auto_arima(train["Sales(in crores)"],start_p=0,start_q=0,max_p=16,max_q=16,m=12,trace=True,Seasonal=True,error_action="warn",stepwise=True)


model1=ARIMA(train["Sales(in crores)"],order=(1,0,1))
res1=model1.fit()

#Forecast for next 12 months
start_index=len(train)
end_index=start_index + 11
Forecast_test=res1.predict(start=start_index,end=end_index)
print(Forecast_test)
import numpy as np
#Evaluate Forecasts
rmse_test=np.sqrt(mean_squared_error(test["Sales(in crores)"],Forecast_test))
print("Test RMSE:%.2f"%rmse_test)
import matplotlib.pyplot as plt
train["Sales(in crores)"].plot(legend=True,label="Train",figsize=(10,6))
test["Sales(in crores)"].plot (legend=True,label="Test")
Forecast_test.plot(legend=True,label="predictionARIMA")         
plt.plot(test["Sales(in crores)"])
plt.plot(forecast_test,color="red")
plt.show()
#SARIMA MODEL
model2=sm.tsa.statespace.SARIMAX(sales["Sales(in crores)"],seasonal_order=(12,1,9,12))
res2=model2.fit()
forecast=res2.predict(start=start_index,end=end_index,dynamic=True)
print(forecast)
sales[["Sales(in crores)",forecast]].plot(figsize=(12,8))
rmse=np.sqrt(mean_squared_error(test["Sales(in crores)"],forecast.dropna()))
print(rmse)
 
#ARIMA model is better than SARIMA model
res.save("/Users/neerajkumar/model.pickle")
from statsmodels.regression.linear_model import OLSResults
model3=OLSResults.load("/Users/neerajkumar/model.pickle")


#Forecast  Sales For Future 12 months


future_forecast=model3.predict(start=len(sales),end=len(sales)+11)
print(future_forecast)
train["Sales(in crores)"].plot(legend=True,label="Train",figsize=(10,6))
test["Sales(in crores)"].plot (legend=True,label="Test")
future_forecast.plot(legend=True,label="FuturePrediction")   

#Demand Prediction
import pandas as pd
sales=pd.read_excel("/Users/neerajkumar/Desktop/SONIA/CementData.xlsx")
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
#EDA

import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(sales["Demand(in tons)"])
sns.boxplot(sales["Demand(in tons)"])

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",tail="both",fold=1.5,variables=["Demand(in tons)"])
winsor.fit_transform(pd.DataFrame(sales["Demand(in tons)"]))
sales["Demand(in tons)"].plot()
plt.hist(sales["Demand(in tons)"])
sns.boxplot(sales["Demand(in tons)"])

#Stationary Check
import statsmodels.api as sm
decomposition=sm.tsa.seasonal_decompose(x=sales["Demand(in tons)"],model="additive",extrapolate_trend="freq",period=2)
fig=decomposition.plot()
plt.show()


from statsmodels.tsa.stattools import adfuller
adftest=adfuller(sales["Demand(in tons)"])
print("p value of adfuller test:",adftest[1])

#It is not stationary


adftest=adfuller(sales["Demand(in tons)"].diff(1).dropna())
print("p value of adfuller test:",adftest[1])

#Now it has become Stationary



train=sales.head(61)
test=sales.tail(12)
tsa_plots.plot_pacf(sales["Demand(in tons)"],lags=12)
tsa_plots.plot_acf(sales["Demand(in tons)"],lags=12)


#ARIMA with AR=1,MA=4
model4=ARIMA(train["Demand(in tons)"],order=(10,1,1))
res4=model4.fit()

#Forecast for next 12 months
start_index=len(train)
end_index=start_index + 11
forecast_test4=res4.predict(start=start_index,end=end_index)
print(forecast_test4)

import numpy as np
#Evaluate Forecasts
rmse_test=np.sqrt(mean_squared_error(test["Demand(in tons)"],forecast_test4))
print("Test RMSE:%.2f"%rmse_test)
import matplotlib.pyplot as plt
train["Demand(in tons)"].plot(legend=True,label="Train",figsize=(10,6))
test["Demand(in tons)"].plot (legend=True,label="Test")
forecast_test4.plot(legend=True,label="predictionARIMA")         
plt.plot(test["Demand(in tons)"])
plt.plot(forecast_test4,color="red")
plt.show()


#AUTOARIMA
#pip install pmdarima
import pmdarima as pm
ar_model=pm.auto_arima(train["Demand(in tons)"],start_p=0,start_q=0,max_p=16,max_q=16,m=12,trace=True,Seasonal=True,error_action="warn",stepwise=True)

model5=ARIMA(train["Demand(in tons)"],order=(2,0,0))
res5=model5.fit()
#AUTOARIMA is giving high error than ARIMA Model

#Forecast for next 12 months
start_index=len(train)
end_index=start_index + 11
forecast_test5=res5.predict(start=start_index,end=end_index)
print(forecast_test5)

import numpy as np
#Evaluate Forecasts
rmse_test=np.sqrt(mean_squared_error(test["Demand(in tons)"],forecast_test5))
print("Test RMSE:%.2f"%rmse_test)
import matplotlib.pyplot as plt
train["Demand(in tons)"].plot(legend=True,label="Train",figsize=(10,6))
test["Demand(in tons)"].plot (legend=True,label="Test")
forecast_test5.plot(legend=True,label="predictionARIMA")         
plt.plot(test["Demand(in tons)"])
plt.plot(forecast_test2,color="red")
plt.show()

#AutoARIMA is not better than ARIMA
#SARIMA MODEL
model6=sm.tsa.statespace.SARIMAX(sales["Demand(in tons)"],seasonal_order=(10,1,1,12))
res6=model6.fit()
sales["forecast1"]=res6.predict(start=start_index,end=end_index,dynamic=True)
print(sales["forecast1"])
sales[["Demand(in tons)","forecast1"]].plot(figsize=(12,8))
rmse=np.sqrt(mean_squared_error(test["Demand(in tons)"],sales["forecast1"].dropna()))
print(rmse)

#SARIMA model is better than ARIMA and AutoARIMA


res6.save("/Users/neerajkumar/model1.pickle")
from statsmodels.regression.linear_model import OLSResults
model7=OLSResults.load("/Users/neerajkumar/model1.pickle")


future_forecast1=model7.predict(start=len(sales),end=len(sales)+11)
print(future_forecast1)
train["Demand(in tons)"].plot(legend=True,label="Train",figsize=(10,6))
test["Demand(in tons)"].plot (legend=True,label="Test")
future_forecast1.plot(legend=True,label="FuturePrediction")   




















