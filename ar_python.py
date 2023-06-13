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
sales_data = pd.read_excel('/****_path_to_where_your_excel_file_is_located_****/Sales 52W.xlsx')

#Convert the 'Date' column to a pandas specific datetime object.
sales_data['Date'] = pd.to_datetime(sales_data['Date'])

#Set the 'Date' column as the dataframe's index
sales_data.set_index('Date', inplace=True)

#Set the frequency of the index to weekly (W)
sales_data.index.freq = 'W-MON'

#Plot histogram of sales data
plt.figure(figsize=(8,5))
plt.hist(sales_data['Sales'],bins=20)
plt.title('Histogram of Sales Data')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

#Compute skewness of 'Sales' data
sales_skewness =skew(sales_data['Sales'])

#Print skewness interpretation
if sales_skewness == 0:
    print('The data is symmetrical.')
elif -1 <= sales_skewness < 0:
    print("The data is left skewed within tolerance. No transformation required. \n")
elif 1 <= sales_skewness:
    print("the data is right skewed within tolerance. No transformation required. \n")
else:
    print("The data exhibits skewness beyond the tolerance range. Transformation might be required. \n")

#ADF test for stationarity
adf_result = adfuller(sales_data['Sales'],regression='ctt')

#print ADF result in tabular form
print("ADF test for stationarity (including seasonal terms):\n")

#define headers and table values for tabulate feature
headers = ['ADF Statistic','p-value','Number of observations']
table = [[adf_result[0],adf_result[1],adf_result[3]]]

print(tabulate(table, headers=headers, tablefmt='fancy_grid', numalign="center", stralign="center"))
print(f'\nCritical Values:\n')
for key, value in adf_result[4].items():
    print(f'{key}: {value}')

#interpret the ADF test result
if adf_result[0] < adf_result[4]['5%']:
    print('\nThe data is stationary (with 95% confidence) and does not exhibit autocorrelation, nor a trend component. \n')
else:
    print('\nThe data is non-stationary and does exhibit autocorrelation and/or a trend component.\n')

#split data into train and test sets
train_sales = sales_data.iloc[:40]
test_sales  = sales_data.iloc[39:]

# Find the best value for lags
best_lags = None
best_score = np.inf
for lags in range(1, 14):
    #fit AR model on train data
    model = AutoReg(train_sales, lags=lags)
    model_fit = model.fit()

    #make prediction on test data
    yhat = model_fit.predict(start=len(train_sales), end=len(train_sales)+len(test_sales)-1)

    #compute result metrics
    mae = mean_absolute_error(test_sales, yhat)
    mse = mean_squared_error(test_sales, yhat)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_sales, yhat)

    #compute score
    score = mae + mse + rmse + (1 - r2)

    #update best score and best lags
    if score < best_score:
        best_score = score
        best_lags = lags

#printing the auto detected optimal parameter of "lags"
print('The most optimum auto-detected "lags" parameter for the given dataset is:', best_lags, '\n')

#fitting AR model on 'train_set' using the auto-detected best 'lags' parameter
model = AutoReg(train_sales, lags=best_lags)
model_fit = model.fit()

#making predictions on 'test_data'
yhat = model_fit.predict(start=len(train_sales), end=len(train_sales)+len(test_sales)-1)

#compute metrics
mae = mean_absolute_error(test_sales, yhat)
mse = mean_squared_error(test_sales, yhat)
rmse = np.sqrt(mse)
r2 = r2_score(test_sales, yhat)

#format the print output as table
#define the data as a list of lists
data=[['MAE', mae],
      ['MSE', mse],
      ['RMSE', rmse],
      ['R-squared', r2]]

#print the data in tabulated format
print(tabulate(data, headers=['Metric','Value'], tablefmt='fancy_grid', numalign='center', stralign='center'))

#make prediction for 13 weeks beyond the existing horizon
forecast_sales = model_fit.predict(start=len(sales_data), end=len(sales_data)+12)

#visualise actual sales qty, train & test data sets and forecasted sales qty
#create a figure with two subplots
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(17,10))

#plot existing sales, train & test data sets values on one subplot
axes[0].set_ylim(0,70)
axes[0].set_ylabel('Sales Qty')
axes[0].set_title('The Autoregression (AR) model breakdown: 52 Wks Sales (Qty) / Train Sales (39 Wks) / Test Sales (13 Wks)')
axes[0].plot(sales_data, label='Actual Sales Qty', color='blue', linewidth=10, alpha=0.4, zorder=0)
axes[0].plot(train_sales, label='Train Sales (39 Wks)', color='black', marker='o', markerfacecolor='black')
axes[0].plot(test_sales, label='Tets Sales (13 Wks)', color='orange', marker='o', markerfacecolor='orange')
axes[0].plot(yhat, label='Predicted Sales Qty vs. Test Sales Qty', color='green',linewidth=4, linestyle='dashed')
axes[0].legend()
axes[0].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=-1)

#plot existing and forecasted sales qty data values on one plot for clarity
axes[1].set_ylim(0,70)
axes[1].set_ylabel('Sales Qty')
axes[1].set_title('The Autoregression (AR) model: 52 Wks Sales (Qty) + 13 Wks Forecasted Sales Qty')
axes[1].plot(sales_data, label='Actual Sales Qty', color='blue', linewidth=10, alpha=0.4, zorder=0)
axes[1].plot(forecast_sales, label='Forecasted Sales Qty (13 Wks)', color='red', linewidth=4, linestyle='dashed')
axes[1].legend()
axes[1].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.4, zorder=-1)

#create a dataframe with the forecasted values
forecasted_13_weeks = np.ceil(pd.DataFrame(forecast_sales, columns=['Sales']))
forecasted_13_weeks['Date'] = ['F.WEEK {}'.format(i+1) for i in range(len(forecast_sales))]

#read the source file again
source_file = '/Users/aj/Desktop/PYTHON/Sales 52W.xlsx'
sales_data = pd.read_excel(source_file)

#append the forecasted values to the botoom of the source file
sales_data = pd.concat([sales_data, forecasted_13_weeks[['Date','Sales']]], axis=0)

#save the updated dataframe to the source excel file
sales_data.to_excel(source_file, index=False)
