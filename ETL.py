import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the source Excel file

# Create a simple DataFrame
df = pd.read_excel('/**** PATH TO THE SOURCE FILE ****/Source file.xlsx')

# Step 2: Perform transformations

#Transformation #1: Convert names to UPPERCASE
df['Product'] = df['Product'].str.upper()

#Transformation #2: Detect outliers based on the quartile method (and isolate them for further review)

Q1 = df['Quantity_Sold'].quantile(0.25)
Q3 = df['Quantity_Sold'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_range= df[(df['Quantity_Sold'] < lower_bound) | (df['Quantity_Sold'] > upper_bound)]
negative_values = df[df['Quantity_Sold'] < 0]

# Move outliers and negative values to a separate DataFrame
outliers_and_negative = pd.concat([outliers_range, negative_values])
df = df.drop(outliers_and_negative.index)

#Transformation #3: Impute empty values in 'Quantity_Sold' column with the column mean

df['Imputed'] = ''
df.loc[df['Quantity_Sold'].isnull(), 'Imputed'] = 'X'
mean_quantity_sold = df['Quantity_Sold'].mean()
df['Quantity_Sold'].fillna(np.ceil(mean_quantity_sold), inplace=True)

# Calculate the 'Total_Revenue' column
df['Total_Revenue'] = df['Quantity_Sold'] * df['Cost_per_unit']

#Transformation #4: Calculate and add the '%_Total_Revenue' column
total_revenue = df['Total_Revenue'].sum()
df['%_Total_Revenue'] = (df['Total_Revenue'] / total_revenue) * 100
df['%_Total_Revenue'] = df['%_Total_Revenue'].round(2)  # Format as float with two decimal places

#Transformation #5: Sort the DataFrame in descending order based on 'Total_Revenue'
df = df.sort_values(by='Total_Revenue', ascending=False)

#Transformation #5 Add the total row to the DataFrame
total_row = pd.DataFrame({'Product': 'Total', 'Quantity_Sold':df['Quantity_Sold'].sum(), 'Total_Revenue': total_revenue}, index=[len(df)])
df = pd.concat([df, total_row]).reset_index(drop=True).fillna('')

#Transformation #6 Reorder columns
df = df[['Product', 'Quantity_Sold', 'Cost_per_unit','Total_Revenue', '%_Total_Revenue', 'Imputed']]

# Step 3: Save the dataframe and chart to the output Excel file 

# Save the DataFrame and chart to an Excel file
output_file = '/*** PATH TO SAVE LOCATION ***/output.xlsx'
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    
    # Save the DataFrame to a sheet named 'Data'
    df.to_excel(writer, sheet_name='Data', index=False)

    # Save the outliers DataFrame to a sheet named 'Outliers'
    outliers_and_negative.to_excel(writer, sheet_name='Outliers', index=False)

    # Get the worksheet
    worksheet = writer.sheets['Data']

    # Create the chart on the same sheet
    chart = writer.book.add_chart({'type': 'column'})
    chart.add_series({
        'categories': '=Data!$A$2:$A$' + str(len(df)),  # Product names as x-axis categories
        'values': '=Data!$D$2:$D' + str(len(df)),  # Exclude the last row (total row)
        'name': 'Total Revenue',  # Set the legend label to 'Total Revenue'
    })
   
    chart.set_title({'name': 'Total Revenue per Product'})
    chart.set_x_axis({'name': 'Product'})
    chart.set_y_axis({'name': 'Total Revenue'})
    chart.set_legend({'position': 'top'})
    chart.set_size({'width': 360, 'height': 288})

    worksheet.insert_chart('H1', chart)

 # Get the worksheet for the 'Outliers' sheet
    worksheet_outliers = writer.sheets['Outliers']

 # Write the outliers DataFrame to the 'Outliers' sheet
    outliers_range.to_excel(writer, sheet_name='Outliers', index=False)

# Print a message to indicate the completion
print("DataFrame and chart saved to output.xlsx.")

