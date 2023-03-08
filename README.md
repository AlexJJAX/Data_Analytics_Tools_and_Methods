# Forecasting in Python

## What is forecasting?

Forecasting is the process of making predictions about future events or trends based on historical data and statistical models. The goal of forecasting is to anticipate what will happen in the future. There are various techniques used for forecasting, including time series analysis, regression analysis, and machine learning algorithms.

# Autoregression (AR) model

## What is Autoregression (AR) model?

The autoregression (AR) method models a time series based on its own past values, also known as lagged values. The AR method is suitable for analysing univariate time series data that does not have trend or seasonal components. It is also important to note that the AR method assumes that the time series is stationary, meaning that its statistical properties such as mean and variance are constant over time. If the time series is not stationary, it may need to be transformed to achieve stationarity before applying the AR method.

## What does the [code](https://github.com/AlexJJAX/Forecasting-in-Python/blob/main/ar_python.py) do?

The code performs time series analysis on a univariate sales dataset. The dataset is read from an Excel file and processed using pandas and numpy libraries. It then plots a histogram of the sales data and computes the skewness of the sales data. The code then performs an augmented Dickey-Fuller test to determine if the data is stationary. The data is then split into a train and test set, and the autoregressive (AR) model is applied to the data. The code determines the best lags parameter for the AR model and fits the model to the train set using the best lags parameter. The model is then used to make predictions for the test set, and evaluation metrics such as mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), and R-squared are computed. Finally, the model is used to forecast sales for the next 13 weeks beyond the existing horizon.
