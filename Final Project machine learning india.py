#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset
df = pd.read_csv("C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\smartphones India.csv")

# 2. Exploración inicial
display(df.head())
display(df.describe())
display(df.info())

# 3. Preprocesamiento
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Características (features) y objetivo (target)
X = df[['resolution_width','resolution_height','avg_rating','battery_capacity', 'internal_memory']]  # Sustituye con las columnas relevantes
y = df['price'] # Sustituye con la columna objetivo 

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
display('Gradient Boosting - Mean Squared Error:', mean_squared_error(y_test, y_pred_gb))
display('Gradient Boosting - R^2 Score:', r2_score(y_test, y_pred_gb))


# In[73]:


errors = y_test - y_pred_gb
plt.figure(figsize=(10, 5))
sns.histplot(errors, kde=True)
plt.title("Errors Distribution - Gradient Boosting")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.xlim(-40000,40000)


# In[35]:


import pandas as pd

# 1. Cargar el CSV en un DataFrame
try:
    df = pd.read_csv('C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\smartphones India.csv')
except FileNotFoundError:
    print("El archivo smartphones.csv no se encontró.")
    raise

# 2. Verificar que las columnas existen
if 'brand_name' not in df.columns or 'price' not in df.columns:
    raise ValueError("El CSV debe contener las columnas 'brand' y 'price'.")

# 3. Asegurarse de que los precios son numéricos
if not pd.api.types.is_numeric_dtype(df['price']):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 4. Eliminar filas con valores de precio no numéricos
df.dropna(subset=['price'], inplace=True)

# 5. Agrupar los datos por marca
grouped = df.groupby('brand_name')

# 6. Calcular la media de los precios por marca
mean_prices = grouped['price'].mean()*0.011

# 7. Mostrar los resultados
print(mean_prices.round(2))


# In[25]:


# Visualización de Resultados del Modelo
plt.figure(figsize=(10, 5))
sns.regplot(x=y_test*0.011, y=y_pred_gb*0.011,scatter_kws={'alpha':0.5})
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Valores Reales vs. Valores Predichos - Gradient Boosting")


# In[26]:


# 2. Preprocesamiento
# Manejar valores nulos
df.fillna(0, inplace=True)

# 3. Filtrar los valores de la columna 'price' entre 9000 y 10000
df_filtered = df[(df['price'] >= 9000) & (df['price'] <= 10000)]

# 4. Ordenar los datos por 'brand_name'
df_sorted = df_filtered.sort_values(by='brand_name')

# Características (features) y objetivo (target)
features = ['resolution_width', 'resolution_height', 'avg_rating', 'battery_capacity', 'internal_memory']
target = 'price'

# 5. Función para predecir valores específicos
def predict_specific_values(model, df, feature_cols):
    predictions = []
    for _, row in df.iterrows():
        input_data = row[feature_cols].values.reshape(1, -1)  # Convertir la fila a un array 2D
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
    return predictions

# Obtener las predicciones para las filas de interés en df_sorted
predictions = predict_specific_values(gb_model, df_sorted, features)

# Añadir las predicciones al DataFrame
df_sorted['predicted_price'] = predictions

# Mostrar la tabla resultante
display(df_sorted[['brand_name', 'resolution_width', 'resolution_height', 'avg_rating', 'battery_capacity', 'internal_memory', 'predicted_price']].sort_values(by='predicted_price', ascending=True))

# Buscar el valor más cercano a 90,000
desired_value = 90000
df_sorted['difference'] = np.abs(df_sorted['predicted_price'] - desired_value)
closest_row = df_sorted.loc[df_sorted['difference'].idxmin()]

# Mostrar la fila con el valor más cercano
display(closest_row[['brand_name', 'resolution_width', 'resolution_height', 'avg_rating', 'battery_capacity', 'internal_memory', 'predicted_price']])

# Definir características (features) y objetivo (target)
features = ['resolution_width', 'resolution_height', 'avg_rating', 'battery_capacity', 'internal_memory']
target = 'price'

# Obtener características (features) y objetivo (target)
X = df[features]

gb_model.fit(X_train, y_train)

# Hacer predicciones sobre todo el conjunto de datos
df['predicted_price'] = gb_model.predict(X)

# Calcular la media de los valores predichos por 'brand_name'
brand_mean_predictions = df.groupby('brand_name')['predicted_price'].mean()

# Calcular la media de los valores de la columna 'price' por 'brand_name'
brand_mean_prices = df.groupby('brand_name')['price'].mean()

mean_df = (brand_mean_predictions-brand_mean_prices).abs().sort_values()

# Combinar ambos resultados en un nuevo DataFrame
result_df = pd.DataFrame({
    'brand_name': brand_mean_predictions.index,
    'predicted_price_mean': brand_mean_predictions.values,
    'price_mean': brand_mean_prices.values,
    'difference': mean_df.values
})

# Mostrar la tabla resultante ordenada por 'brand_name'
display(result_df.sort_values(by='difference',ascending=True))


# In[27]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
battery_capacity=3110.0
internal_memory=64
avg_rating=7.3
resolution_height=1792
resolution_width=828

new_data=pd.DataFrame({
    'resolution_width':[resolution_width],'resolution_height':[resolution_height],'avg_rating':[avg_rating],'battery_capacity':[battery_capacity],'internal_memory':[internal_memory]
})
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización de los datos
    ('gb', GradientBoostingRegressor(random_state=42))
])
predicted_price=gb_model.predict(new_data)
print("predicted_price :" ,predicted_price)


# In[28]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume df is already defined with the following columns:
columns = [
    'price', 'avg_rating', '5G_or_not', 'num_cores', 
    'processor_speed', 'battery_capacity', 'fast_charging_available', 
    'internal_memory', 'screen_size', 'refresh_rate', 'num_rear_cameras', 'primary_camera_rear', 'primary_camera_front', 
    'extended_memory_available', 'resolution_height', 'resolution_width'
]

# Select only the specified columns
df= df[columns]

# Compute the correlation matrix
corr = np.abs(df.corr())

# Set up the mask for the upper triangle
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=corr, cmap=cmap)

plt.show()


# In[29]:


df_selected = df[['resolution_height', 'avg_rating']]

print(df_selected.head(20))

print(df_selected.describe())


# In[77]:


import pandas as pd

# 1. Cargar el CSV en un DataFrame
try:
    df = pd.read_csv('C:\\Users\\gines\\Desktop\\Projectfolder\\Finalproject\\smartphones India.csv')
except FileNotFoundError:
    print("El archivo smartphones.csv no se encontró.")
    raise
    
df.fillna(0, inplace=True)
    
# Definir características (features) y objetivo (target)
display(df.head())
features = ['resolution_width', 'resolution_height', 'avg_rating', 'battery_capacity', 'internal_memory']
target = 'price'

# Obtener características (features) y objetivo (target)
X = df[features]

gb_model.fit(X_train, y_train)

# Hacer predicciones sobre todo el conjunto de datos
df['predicted_price'] = gb_model.predict(X)

# Calcular la media de los valores predichos por 'brand_name'
brand_mean_predictions = df.groupby('brand_name')['predicted_price'].mean()*0.011

# Calcular la media de los valores de la columna 'price' por 'brand_name'
brand_mean_prices = df.groupby('brand_name')['price'].mean()*0.011

brand_mean_prices=brand_mean_prices.round(2)
brand_mean_predictions=brand_mean_predictions.round(2)
mean_df=(brand_mean_predictions-brand_mean_prices)

# Combinar ambos resultados en un nuevo DataFrame
result_df = pd.DataFrame({
    'brand_name': brand_mean_predictions.index,
    'predicted_price_mean': brand_mean_predictions.values,
    'price_mean': brand_mean_prices.values,
    'difference': mean_df.values
})

# Mostrar la tabla resultante ordenada por 'brand_name'
display(result_df.sort_values(by='difference',ascending=True))


# In[81]:


graph_df = result_df.sort_values(by='difference',ascending=True)
plt.figure(figsize=(12, 8))
sns.barplot(x='difference', y='brand_name', data=graph_df, palette='viridis')
plt.title('Diferencia Media entre Predicciones y Precios Reales por Marca')
plt.xlabel('Diferencia Media (Absoluta)')
plt.ylabel('Marca')
plt.show()


# In[ ]:




