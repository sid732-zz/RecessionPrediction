# inflation

import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from  matplotlib import pyplot as plt
import pandas as pd
#df = pd.read_csv('inflation-rate.csv')

df = quandl.get("RATEINF/INFLATION_USA",authtoken='SJ5_m2aJeE3NNbF8gZeN') # FRED/GDP datset for gdp
# take a look on the data

#df = pd.read_csv('usa-Inflation.csv')

print(df)

df = df[['Value']]
print(df.head())

forecast_out = int(input("Enter the numbers of inflation prediction:"))

#forecast_out = 3
df['prediction'] = df[['Value']].shift(-forecast_out)
print(df.head())
print(df.tail())

X = np.array(df.drop(['prediction'],1))
X = X[:-forecast_out]
print(X)

y = np.array(df['prediction'])
y = y[:-forecast_out]
print(y)


x_train, x_test,y_train, y_test = train_test_split(X,y,test_size=0.2)

svr_rbf = SVR(kernel='rbf',C=1e3, gamma=0.1)
svr_rbf.fit(x_train,y_train)

svm_confidence = svr_rbf.score(x_test, y_test)
print("svr confidence:",svm_confidence)

lr = LinearRegression()
lr.fit(x_train,y_train)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence:",lr_confidence)

x_forecast = np.array(df.drop(['prediction'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print("lr prediction",lr_prediction)

svr_prediction = svr_rbf.predict(x_forecast)
print("svr prediction: ",svr_prediction)
