#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd

# Define the CSV file path with double backslashes for Windows
csv_file_path = "C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\Sales.csv"


# Assuming your CSV file exists and is in the correct format
try:
  # Attempt to read the CSV file using pandas.read_csv
  df = pd.read_csv(csv_file_path)
  print(df)  # Print the DataFrame
except FileNotFoundError:
  print("Error: CSV file not found. Please check the file path.")


#Convert all column names to lowercase for consistency
def lowercase_column_names(df):
    """
    Conver all column names of a DataFrame to lowercase.

    Parameters:
    df: The DataFrame whose column names need to be converted

    Returns:
    df: The DataFrame with all column names in lowercase
    """
    df = df.rename(columns=str.lower)
    return df

df = lowercase_column_names(df)
df.head()
df.columns=df.columns.str.replace(' ','_')

#display any missing values in the dataset
df.isnull().sum()

df_cleaned = df.dropna(subset=['memory'])
df['memory'].fillna(0, inplace=True)
df_cleaned = df.dropna(subset=['rating'])
df['rating'].fillna(0, inplace=True)
df_cleaned = df.dropna(subset=['storage'])
df['storage'].fillna(0, inplace=True)
df.isnull().sum()
print(df)

group = df.groupby('colors')
mean = group['rating'].mean().round(2)

print(mean)

df['profit margen']=df['original_price']*0.011 -df['selling_price']*0.011
print(df[['brands', 'models', 'selling_price', 'original_price', 'profit margen']])

mean_profit=df.groupby('brands')['profit margen'].mean().round(2).head(6).sort_values(ascending=False)
print(mean_profit)

import pandas as pd

# Define the CSV file path (adjust as needed)
csv_file_path = "C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\Sales.csv"


def calculate_discount(df):
    """Calculates discount amount and percentage for each smartphone in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing smartphone data.

    Returns:
        pandas.DataFrame: The DataFrame with additional columns for discount amount and percentage.
    """

    df['discount'] = df['original_price'] - df['selling_price']
    df['discount_percentage'] = (df['discount'] / df['original_price']) * 100
    return df

# Handle missing values in memory, rating, and storage columns
for col in ['memory', 'rating', 'storage']:
    df[col].fillna(0, inplace=True)  # Fill with 0 or a more appropriate value

# Calculate discount and discount percentage
df = calculate_discount(df.copy())  # Avoid modifying original DataFrame

# Group by brands and sort by profit margin (descending)
top_brands_profit_margins = (
    df.groupby('brands')['profit margen']
    .mean()
    .round(2)
    .sort_values(ascending=False)
    .head(5)
)

# Get top 5 brands with highest and lowest discount percentages
top_discount_brands = (
    df.groupby('brands')['discount_percentage']
    .mean()
    .round(2)
    .sort_values(ascending=False)
    .head(5)
)
lowest_discount_brands = (
    df.groupby('brands')['discount_percentage']
    .mean()
    .round(2)
    .sort_values(ascending=True)
    .head(5)
)

# Print results
print("\nTop 5 Brands with Highest Profit Margins:")
print(top_brands_profit_margins)

print("\nTop 5 Brands with Highest Discount Percentages:")
print(top_discount_brands)

print("\nTop 5 Brands with Lowest Discount Percentages:")
print(lowest_discount_brands)

group for brands and storage and mean prices
storage_sorted=df.sort_values(by=['brands','storage','original_price'],ascending=[True,False,True]).groupby('brands')
print(storage_sorted[['brands','storage','original_price']].head(5))


# In[ ]:




