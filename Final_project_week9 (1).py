#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[49]:


#load dataset

csv_file_path = "C:\\Users\\gines\\Downloads\\smartphones.csv"

# Read the data from the CSV file
df = pd.read_csv(csv_file_path)

# Print the DataFrame (optional)
display(df)


# In[ ]:


#Data Cleaning


# In[11]:


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


# In[12]:


df = lowercase_column_names(df)
df.head()


# In[13]:


#display any missing values in the dataset
df.isnull().sum()


# In[36]:


#Defining a function to clean the columns containing missing values

def missingvalues_df(df):

  # Check data types before imputation
  if pd.api.types.is_numeric_dtype(df['processor_speed']):
    df['avg_rating'].fillna(df['processor_speed'].mean(), inplace=True)
  else:
    display("Warning: processor_speed is not numerical, using mode for avg_rating")
    df['avg_rating'].fillna(df['avg_rating'].mode().iloc[0], inplace=True)

  if pd.api.types.is_numeric_dtype(df['os']):
    df['battery_capacity'].fillna(df['os'].mean(), inplace=True)
  else:
    display("Warning: os is not numerical, using mode for battery_capacity")
    df['battery_capacity'].fillna(df['battery_capacity'].mode().iloc[0], inplace=True)

  # Consider alternative to dropping rows (commented out)
  # df.dropna(subset=['fast_charging', 'avg_rating'], inplace=True)

  return df  # Return statement inside the function

clean_df = missingvalues_df(df.copy())


display(clean_df)


# In[39]:


#Defining a function to clean the columns containing missing values

def missingvalues_df(df):

  if pd.api.types.is_numeric_dtype(df['processor_speed']):
    df['avg_rating'].fillna(df['processor_speed'].mean(), inplace=True)
  else:
    display("Warning: processor_speed is not numerical, using mode for avg_rating")
    df['avg_rating'].fillna(df['avg_rating'].mode().iloc[0], inplace=True)

  if pd.api.types.is_numeric_dtype(df['os']):
    df['battery_capacity'].fillna(df['os'].mean(), inplace=True)
  else:
    display("Warning: os is not numerical, using mode for battery_capacity")
    df['battery_capacity'].fillna(df['battery_capacity'].mode().iloc[0], inplace=True)

  df['primary_camera_front'].fillna(df['primary_camera_front'].mode().iloc[0], inplace=True)

  return df

clean_df = missingvalues_df(df.copy())  

display(clean_df)


# In[35]:


df_smartphones = missingvalues_df(df)


# In[40]:


#checking again whether there are any remaining missing values
df_smartphones.isnull().sum()


# In[41]:


df_smartphones.head()


# In[ ]:


## Exploratory Data Analysis


# In[44]:


#Generate a color palette with the same number of uniques values for 'make'
num_colors = df_smartphones['brand_name'].nunique()
palette = sns.color_palette("husl", num_colors)

#get the value counts of the 'brand name' columns and sort by count
order = df_smartphones['brand_name'].value_counts().index

#Create a countplot for each car brand in the dataset
plt.figure(figsize = (20,15))
sns.countplot(y=df_smartphones['brand_name'], order = order, palette = palette)
plt.title("Brands smartphones most sold")
plt.show()


# In[46]:


#Generate a color palette with the same number of uniques values for 'year'
num_colors_year = df_smartphones['avg_rating'].nunique()
palette_year = sns.color_palette("husl", num_colors_year)

#Create a countplot for each year in the dataset
plt.figure(figsize = (20,15))
sns.countplot(x=df_smartphones['avg_rating'], palette = palette_year)
plt.title("Smartphones Average Rating")
plt.show()


# In[2]:


df_smartphones = pd.DataFrame(data)


# In[4]:


import pandas as pd

# Define the file path (replace with your actual file path)
data_file = "C:\\Users\\gines\\Downloads\\smartphones.csv"

# Read data from CSV file (assuming the file has headers)
df_smartphones = pd.read_csv(data_file)


# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your smartphone data in a DataFrame called 'df_smartphones'

# Truncate long rating labels (modify truncation length as needed)
df_smartphones['truncated_rating'] = df_smartphones['avg_rating'].apply(
    lambda x: str(x)[:5] if len(str(x)) > 5 else str(x)
)

# Generate a color palette with a reasonable number of colors (adjust as needed)
num_colors_year = df_smartphones['avg_rating'].nunique() // 2
palette_year = sns.color_palette("husl", num_colors_year)

# Create a countplot with truncated labels and moderate rotation
plt.figure(figsize=(22, 15))  # Adjust width as needed
sns.countplot(x=df_smartphones['truncated_rating'], palette=palette_year)
plt.xticks(rotation=30)  # Moderate rotation

# Add a title and display the plot
plt.title("Smartphones Average Rating")
plt.xlabel("Truncated Average Rating")  # Adjust label text if needed
plt.show()


# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your smartphone data in a DataFrame called 'df_smartphones'

# Ensure 'brand_name' has string values (avoid mixed data types)
df_smartphones['brand_name'] = df_smartphones['brand_name'].astype(str)

# Count smartphone occurrences by brand
brand_counts = df_smartphones['brand_name'].value_counts()

# Sort brands by count in descending order (optional)
brand_counts = brand_counts.sort_values(ascending=False)

# Create a horizontal bar chart
plt.figure(figsize=(18, 10))  # Adjust size as needed
sns.barplot(x=brand_counts, y=brand_counts.index, palette="husl")

# Increase font sizes for readability
plt.xlabel("Number of Smartphones", fontsize=16)
plt.ylabel("Brand Name", fontsize=16)
plt.title("Smartphones Average Rating by Brand", fontsize=18)
plt.tick_params(labelsize=14)  # Set font size for labels

# Optional: Rotate x-axis labels if needed (experiment with angles)
# plt.xticks(rotation=45)

# Display the plot
plt.show()


# In[13]:


#Generate a color palette with the same number of uniques values for 'engine fuel type'
num_colors_type = df_smartphones['processor_brand'].nunique()
palette_type = sns.color_palette("husl", num_colors_type)

#get the value counts of the 'engine fuel type' columns and sort by count
order_type = df_smartphones['processor_brand'].value_counts().index

#Create a countplot for each fuel type in the dataset
plt.figure(figsize = (18,10))
sns.countplot(y=df_smartphones['processor_brand'], order = order_type, palette = palette_type)
plt.show()


# In[17]:


# Assuming your data is stored in a pandas DataFrame named 'df'

# Select the "processor_speed" column and display all rows
processor_speeds = df_smartphones["processor_speed"]
print(processor_speeds)


# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your smartphone data in a DataFrame called 'df_smartphones'

# Ensure 'processor_speed' has numeric values (avoid mixed data types)
df_smartphones['processor_speed'] = pd.to_numeric(df_smartphones['processor_speed'], errors='coerce')

# Generate a color palette with a reasonable number of colors (adjust as needed)
num_colors_processor_speed = df_smartphones['processor_speed'].nunique()
palette_processor_speed = sns.color_palette("husl", num_colors_processor_speed)

# Get value counts of the 'processor_speed' column and sort by count
order_processor_speed = df_smartphones['processor_speed'].value_counts().index

# Create a countplot with sorted processor speeds on x-axis
plt.figure(figsize=(18, 10))
ax = sns.countplot(
    x=df_smartphones['processor_speed'], order=order_processor_speed, palette=palette_processor_speed
)

# Rotate x-axis labels for better readability (adjust rotation angle if needed)
plt.xticks(rotation=45)

# Increase font sizes for readability (optional)
plt.xlabel("Processor Speed", fontsize=14)
plt.ylabel("Number of Smartphones", fontsize=14)
plt.title("Smartphones by Processor Speed", fontsize=16)
plt.tick_params(labelsize=12)  # Set font size for x-axis tick labels

# Adjust space for x-axis labels (experiment with 'tight_layout' or 'subplots_adjust')
plt.tight_layout()  # One approach to adjust spacing

# Display the plot
plt.show()


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Assuming processor speed is numerical

# Assuming you have your smartphone data in a DataFrame called 'df_smartphones'

# Ensure 'processor_speed' has numeric values
df_smartphones['processor_speed'] = pd.to_numeric(df_smartphones['processor_speed'], errors='coerce')

# Create a histogram
plt.figure(figsize=(12, 6))
plt.hist(df_smartphones['processor_speed'], bins=10, edgecolor='black')

# Add labels and title
plt.xlabel("Processor Speed", fontsize=14)
plt.ylabel("Number of Smartphones", fontsize=14)
plt.title("Distribution of Processor Speeds", fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.show()


# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have your smartphone data in a DataFrame called 'df_smartphones'

# Assuming 'screen_size' is numerical
plt.figure(figsize=(10, 6))
sns.kdeplot(df_smartphones['screen_size'], shade=True)  # Or use plt.hist for histogram

# Add labels and title
plt.xlabel("Screen Size (inches)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Distribution of Screen Sizes", fontsize=16)

# Display the plot
plt.show()


# In[30]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named 'df_smartphones'

# Ensure 'fast_charging_available' is a boolean or categorical data type
df_smartphones['fast_charging_available'] = df_smartphones['fast_charging_available'].astype(bool)  # Assuming boolean type

# Group data by brand and calculate counts for fast charging availability
fast_charging_counts = df_smartphones.groupby('brand_name')['fast_charging_available'].value_counts().unstack(fill_value=0)

# Create a stacked bar chart
plt.figure(figsize=(12, 8))
fast_charging_counts.plot(kind='bar', stacked=True, colormap='Set2')

# Add labels and title
plt.xlabel("Smartphone Brand", fontsize=14)
plt.ylabel("Number of Smartphones", fontsize=14)
plt.title("Fast Charging Availability by Brand", fontsize=16)
plt.legend(title="Fast Charging", labels=['Yes', 'No'], loc='upper right', bbox_to_anchor=(1.2, 1))  # Adjust legend position if needed

# Rotate x-axis labels for better readability (adjust rotation angle if needed)
plt.xticks(rotation=45)

# Increase font sizes for readability (optional)
plt.tick_params(labelsize=12)

# Display the plot
plt.show()


# In[47]:


# Assuming your data is stored in a DataFrame named 'df_smartphones'

# Calculate mean availability (Yes/No) per brand
availability_mean = df_smartphones.groupby('brand_name')['fast_charging_available'].mean().reset_index()

# Sort by mean availability (descending)
availability_mean_sorted = availability_mean.sort_values(by='fast_charging_available', ascending=False)

# Plot the mean availability per brand (horizontal)
plt.figure(figsize=(12, 10))  # Adjusted figure size for horizontal chart
sns.barplot(data=availability_mean_sorted, x='fast_charging_available', y='brand_name', orient='h')
plt.title("Fast Charging Availability by Brand", fontsize=16)  # Adjusted title

# Adjust labels (optional)
plt.xlabel("Mean Availability", fontsize=14)  # Adjusted label
plt.ylabel("Brand Name", fontsize=14)  # Adjusted label

# Increase font sizes for readability (optional)
plt.tick_params(labelsize=12)

plt.show()


# In[33]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame named 'df_smartphones'

# Ensure 'fast_charging_available' is a boolean or categorical data type
df_smartphones['fast_charging_available'] = df_smartphones['fast_charging_available'].astype(bool)  # Assuming boolean type

# Group data by brand and calculate counts for fast charging availability
fast_charging_counts = df_smartphones.groupby('brand_name')['fast_charging_available'].value_counts().unstack(fill_value=0)

# Create a horizontal stacked bar chart
plt.figure(figsize=(12, 10))  # Adjust size as needed
fast_charging_counts.plot(kind='barh', stacked=True, colormap='Set2')

# Add labels and title
plt.xlabel("Number of Smartphones", fontsize=14)
plt.ylabel("Brand Name", fontsize=14)  # Y-axis now shows brands
plt.title("Fast Charging Availability by Brand", fontsize=16)
plt.legend(title="Fast Charging", labels=['Yes', 'No'], loc='upper right', bbox_to_anchor=(1.2, 1))  # Adjust legend position if needed

# Increase font sizes for readability (optional)
plt.tick_params(labelsize=12)

# Rotate x-axis labels for better readability (optional)


# In[13]:


import pandas as pd

# Define the CSV file path with double backslashes for Windows
csv_file_path2 = "C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\smartphones.csv"
try:
  # Attempt to read the CSV file using pandas.read_csv
  df2 = pd.read_csv(csv_file_path2)
  display(df2)  # Print the DataFrame
except FileNotFoundError:
  display("Error: CSV file not found. Please check the file path.")

# Optional error handling for other potential issues like incorrect data format


# In[12]:


import pandas as pd

# Define the CSV file path with double backslashes for Windows
csv_file_path = "C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\Sales.csv"


# Assuming your CSV file exists and is in the correct format
try:
  # Attempt to read the CSV file using pandas.read_csv
  df = pd.read_csv(csv_file_path)
  display(df)  # Print the DataFrame
except FileNotFoundError:
  display("Error: CSV file not found. Please check the file path.")

# Optional error handling for other potential issues like incorrect data format


# In[14]:


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


# In[15]:


df = lowercase_column_names(df)
df.head()


# In[16]:


#display any missing values in the dataset
df.isnull().sum()


# In[23]:


df_cleaned = df.dropna(subset=['memory'])
df['memory'].fillna(0, inplace=True)
df_cleaned = df.dropna(subset=['rating'])
df['rating'].fillna(0, inplace=True)
df_cleaned = df.dropna(subset=['storage'])
df['storage'].fillna(0, inplace=True)
df.isnull().sum()


# In[27]:


import pandas as pd

# Define the CSV file path with double backslashes for Windows
csv_file_path = "C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\Sales.csv"

df_Sales = pd.DataFrame(data)


# In[28]:


group = df_Sales.groupby(colors)
mean = grouped['rating'].mean().round(2)


# In[ ]:




