
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="IA Predictora de Precios")
st.header('Predicción de Precios de Coches con Machine Learning')

# 1. Cargar y limpiar datos
try:
    df = pd.read_csv('vehicles_us.csv')
    # Limpiamos nulos en las columnas clave para el modelo
    df_clean = df[['price', 'odometer']].dropna()
except FileNotFoundError:
    st.error("Archivo no encontrado.")
    st.stop()

# 2. Entrenar el modelo (esto sucede al cargar la app)
# X = Kilometraje (Entrada), y = Precio (Objetivo)
X = df_clean[['odometer']]
y = df_clean['price']
model = LinearRegression()
model.fit(X, y)

# 3. Interfaz de Predicción
st.subheader('¿Cuánto debería costar un coche?')
km_input = st.slider('Selecciona el kilometraje (odómetro)', 
                     min_value=0, 
                     max_value=400000, 
                     value=100000)

# Realizar la predicción
prediction = model.predict([[km_input]])

st.metric(label="Precio Estimado", value=f"${prediction[0]:,.2f}")

# 4. Visualización de la lógica
if st.checkbox('Mostrar gráfico de tendencia'):
    fig = px.scatter(df_clean.sample(1000), x="odometer", y="price", 
                     trendline="ols", title="Relación Kilometraje vs Precio")
    st.plotly_chart(fig)