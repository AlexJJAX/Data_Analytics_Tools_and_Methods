#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, boxcox
from tabulate import tabulate

#read the excel file
sales_data = pd.read_excel('/****_path_to_the_excel_file_****/Sales 52W.xlsx')

# Convert the 'Date' column to a pandas datetime object
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

# Set the 'Date' column as the DataFrame's index
sales_data.set_index('Date', inplace=True)

# Set the frequency of the index to weekly (W)
sales_data.index.freq = 'W-MON'

# Plot histogram of sales data
plt.figure(figsize=(8, 5))
plt.hist(sales_data['Sales'], bins=20)
plt.title('Histogram of Sales Data')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Compute skewness of sales data
sales_skewness = skew(sales_data['Sales'])

# Print skewness 
if sales_skewness == 0:
    print("The data is symmetrical. ")
elif -1 <= sales_skewness < 0:
    print("The data is left skewed within tolearnce. No transformation required. \n")
elif 1 <= sales_skewness:
    print("The data is right skewed within tolearnce.No transformation required. \n")

# ADF test for stationarity
adf_result = adfuller(sales_data['Sales'], autolag='AIC', regression='ct')
print("Augmented Dickey-Fuller (ADF) test for stationarity (including seasonal terms): \n")
headers = ['ADF Statistic', 'p-value', 'Lags used', 'Number of observations']
table = [[adf_result[0], adf_result[1], adf_result[2], adf_result[3]]]
print(tabulate(table, headers=headers, tablefmt='fancy_grid', numalign="center", stralign="center"))
print(f'\nCritical Values:\n')
for key, value in adf_result[4].items():
    print(f'{key}: {value}')

# Interpret and print the ADF test result
if adf_result[0] < adf_result[4]['5%']:
    print('\nThe data is STATIONARY (with 95% confidence), and does not exhibit autocorrelation, nor a trend component. \n')
else:
    print('\nThe data is NON-STATIONARY, and does exhibit autocorrelation and/or a trend component. \n')

# Split data into train and test sets
train_sales = sales_data.iloc[:40]
test_sales = sales_data.iloc[39:]

# Find the best value for lags
best_lags = None
best_score = np.inf
for lags in range(1, 14):
    # Fit AR model on train data
    model = AutoReg(train_sales, lags=lags)
    model_fit = model.fit()

    # Make prediction on test data
    yhat = model_fit.predict(start=len(train_sales), end=len(train_sales)+len(test_sales)-1)

    # Compute metrics
    mae = mean_absolute_error(test_sales, yhat)
    mse = mean_squared_error(test_sales, yhat)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_sales, yhat)

    # Compute score
    score = mae + mse + rmse + (1 - r2)

    # Update best score and best lags
    if score < best_score:
        best_score = score
        best_lags = lags

#printing the best parameter of "lags"
print('The most optimum auto-detected "lags" parameter for the given dataset is:', best_lags, '\n')

# Fiting AR model on train set using the best lags
model = AutoReg(train_sales, lags=best_lags)
model_fit = model.fit()

# Make prediction on test data
yhat = model_fit.predict(start=len(train_sales), end=len(train_sales)+len(test_sales)-1)

# Compute metrics
mae = mean_absolute_error(test_sales, yhat)
mse = mean_squared_error(test_sales, yhat)
rmse = np.sqrt(mse)
r2 = r2_score(test_sales, yhat)

# Format the print output as table
# Define the data as a list of lists
data = [["MAE", mae],
        ["MSE", mse],
        ["RMSE", rmse],
        ["R-squared", r2]]

# Print the data in a table
print(tabulate(data, headers=["Metric", "Value"], tablefmt='fancy_grid', numalign="center", stralign="center"))

# Make prediction for 13 weeks beyond the existing horizon
forecast_sales = model_fit.predict(start=len(sales_data), end=len(sales_data)+12)

# Create a figure with two subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(17, 10))

# Plot existing, train and test sales data values on one subplot
axes[0].set_ylim(0,70)
axes[0].set_ylabel('Sales qty')
axes[0].set_title('The autoregression (AR) method model breakdown: 52 Wks Sales / Train Set (39 Wks) / Test Sales (13 Wks)')
axes[0].plot(sales_data, label='Actual Sales', color='blue', linewidth=10, alpha=0.4, zorder=0)
axes[0].plot(train_sales, label='Train Sales (39 Wks)', color='black',  marker='o', markerfacecolor='black')
axes[0].plot(test_sales, label='Test Sales (13 Wks)', color='orange',  marker='o', markerfacecolor='orange')
axes[0].plot(yhat, label='Predicted Sales vs. Test Sales', color='green', linewidth=4, linestyle='dashed')
axes[0].legend()
axes[0].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=-1)

# Plot existing and forecasted sales data values on one subplot
axes[1].set_ylim(0,70)
axes[1].set_title('The autoregression (AR) method model / 52 Wks Sales + 13 Wks Forecast')
axes[1].plot(sales_data, label='Actual Sales (52 Wks)', color='blue', linewidth=10, alpha=0.4, zorder=0)
axes[1].plot(forecast_sales, label='Forecasted Sales Qty (13 Wks)', color='red', linewidth=4, linestyle='dashed')
axes[1].legend()
axes[1].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=-1)

# Display the subplots
plt.show()
