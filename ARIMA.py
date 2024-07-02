"""
Created with assistance from ChatGPT
ARIMA (AutoRegressive Integrated Moving Average)
Time series forecasting
Three Components:
Autoregression (AR)
Differencing (I)
Moving Average (MA)

Autoregression
The dependency between an observation and a number of lagged obervations.
yt​=c+ϕ1​yt−1​+ϕ2​yt−2​+…+ϕp​yt−p​+ϵt​
yt​ is the value at time t
c is a constant
ϕ1,ϕ2,…,ϕpϕ1​,ϕ2​,…,ϕp​ are parameters
ϵt is white noise

Differencing
Subtracting the previous observation from the current observation to make the series stationary.
yt′​=yt​−yt−1​

Moving Average (MA)
The dependency between an observation and a residual error from a moving average applied to lagged observations.
yt​=c+ϵt​+θ1​ϵt−1​+θ2​ϵt−2​+…+θq​ϵt−q​
θ1​,θ2​,…,θq​ are parameters
ϵt​ is white noise

Steps
Load data
Make series stationary
Fit the model
Forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def difference (_data, _interval = 1):
    result = [_data[idx] - _data[idx - _interval] for idx in range(_interval, len(_data))]
    return np.array(result)

def fit_arima_model (_data, _p, _d, _q):
    diff_data = np.array(_data)
    for _ in range(_d):
        diff_data = difference(diff_data)
    X_ar = np.array([diff_data[i - _p : i] for i in range(_p, len(diff_data))])
    y = diff_data[_p:]
    XTX_ar = np.dot(X_ar.T, X_ar)
    XTy_ar = np.dot(X_ar.T, y)
    ar_coefficients = np.dot(np.linalg.inv(XTX_ar), XTy_ar)
    ar_coefficients = ar_coefficients.flatten()
    residuals = y - np.dot(X_ar, ar_coefficients)
    X_ma = np.array([residuals[idx - _q : idx] for idx in range(_q, len(residuals))])
    y_ma = residuals[_q:]
    XTX_ma = np.dot(X_ma.T, X_ma)
    XTy_ma = np.dot(X_ma.T, y_ma)
    ma_coefficients = np.dot(np.linalg.inv(XTX_ma), XTy_ma)
    ma_coefficients = ma_coefficients.flatten()
    return ar_coefficients, ma_coefficients

def forecast_arima (_data, _ar_coeffs, _ma_coeffs, _p, _d, _q, steps = 1):
    forecast = []
    data = list(_data)
    residuals = [0] * _q
    for _ in range(steps):
        ar_part = np.dot(_ar_coeffs, data[-_p:])
        ma_part = np.dot(_ma_coeffs, residuals[-_q:]) if len(residuals) >= _q else 0
        next_value = ar_part + ma_part
        forecast.append(next_value)
        data.append(next_value)
        residuals.append(next_value - np.dot(_ma_coeffs, residuals[-_q:]))
    last_date = pd.to_datetime(_data.index[-1])
    forecast_dates = pd.date_range(start = last_date + pd.Timedelta(days = 1), periods=steps)
    forecast_prices = [prices[-1]]
    for value in forecast:
        forecast_prices.append(forecast_prices[-1] + value)
    del forecast_prices[0]
    return forecast_dates, forecast, forecast_prices
    
data = pd.read_csv("F.csv")
data["Date"] = pd.to_datetime(data["Date"])
data.set_index("Date", inplace = True)
prices = data["Close"]
# Autoregressive order
"""
Definition: p represents the number of lagged observations included in the model.
Increase p: Include more past observations in the model.
Decrease p: Use fewer past observations.
"""
# Integrated order
"""
Definition: d represents the number of times that the raw observations are differenced to make the time series stationary.
Increase d: Perform more differencing to achieve stationarity.
Decrease d: Use fewer differences or even zero if the series is already stationary.
"""
# Moving Average Order
"""
Definition: q represents the number of lagged forecast errors in the prediction equation.
Increase q: Include more lagged forecast errors in the model.
Decrease q: Use fewer lagged forecast errors.
"""
p, d, q = 2, 1, 2
ar_coeffs, ma_coeffs = fit_arima_model(prices.values, p, d, q)
forecast_dates, forecast_values, forecast_prices = forecast_arima(prices, ar_coeffs, ma_coeffs, p, d, q, steps = 100)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Value': forecast_values, 'Forecasted Price' : forecast_prices})
print(forecast_df)
                         
