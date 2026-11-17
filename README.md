# emisi-nes-CO2-Colombia-
# PROYECTO FINAL: ANÁLISIS DE EMISIONES ANUALES DE CO₂ DE COLOMBIA
# Integrantes: Erika Cortes, Ximena Celix, Isabella Cardales
# Curso: Introducción a Ciencias de Datos
# Objetivo: Limpiar, visualizar datos y ajustar un modelo de regresión lineal entre el año y las emisiones de CO₂

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"

# Cargar dataset
datos_originales = pd.read_csv(url)

# Mostrar dimensiones y columnas
print("Dimensiones del archivo:", datos_originales.shape)
print("Columnas disponibles:", datos_originales.columns.tolist())

# Seleccionar columnas necesarias
columnas = ["country", "year", "co2", "co2_per_capita", "population"]
faltantes = [col for col in columnas if col not in datos_originales.columns]

if faltantes:
    raise ValueError(f"Columnas faltantes: {faltantes}")

# Filtrar solo Colombia
datos = datos_originales.loc[datos_originales["country"].eq("Colombia"), columnas].copy()
filas_iniciales = len(datos)

# Convertir a valores numéricos
for col in ["year", "co2", "co2_per_capita", "population"]:
    datos[col] = pd.to_numeric(datos[col], errors="coerce")

# Eliminar nulos, duplicados y valores negativos
datos = datos.dropna(subset=["year", "co2"])
datos = datos[datos["co2"] >= 0]
datos = datos.drop_duplicates(subset="year")
datos["year"] = datos["year"].astype(int)

# Limitar análisis desde el año 1990
datos = datos[datos["year"] >= 1990].sort_values("year").reset_index(drop=True)

if datos.empty:
    raise ValueError("No hay registros válidos para Colombia")

# Resumen
print(f"Filas iniciales Colombia: {filas_iniciales}")
print(f"Filas después de limpieza: {len(datos)}")
print(f"Periodo analizado: {datos['year'].min()} - {datos['year'].max()}")
display(datos.head())
display(datos[["co2", "co2_per_capita", "population"]].describe())

datos_per_capita = datos.dropna(subset=["co2_per_capita"])

if not datos_per_capita.empty:
    plt.figure(figsize=(11,5))
    sns.lineplot(data=datos_per_capita, x="year", y="co2_per_capita", marker="o")
    plt.title("Emisiones de CO₂ por habitante en Colombia")
    plt.xlabel("Año")
    plt.ylabel("Toneladas CO₂ por habitante")
    plt.tight_layout()
    plt.show()
else:
    print("Sin datos per‑cápita para graficar")

plt.figure(figsize=(11,5))
sns.lineplot(data=datos, x="year", y="co2", marker="o")
plt.title("Emisiones anuales totales de CO₂ de Colombia")
plt.xlabel("Año")
plt.ylabel("Emisiones CO₂ (millones toneladas)")
plt.tight_layout()
plt.show()



if len(datos) < 5:
    raise ValueError("Se necesitan mínimo 5 registros")

# Variables
X = datos[["year"]]
y = datos["co2"]

# División cronológica 80 % entrenamiento / 20 % prueba
punto_corte = int(len(datos)*0.8)
X_train, X_test = X.iloc[:punto_corte], X.iloc[punto_corte:]
y_train, y_test = y.iloc[:punto_corte], y.iloc[punto_corte:]

# Crear y entrenar modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predicciones
predicciones = modelo.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, predicciones)
rmse = mean_squared_error(y_test, predicciones)**0.5
r2 = r2_score(y_test, predicciones)

# Imprimir resultados
print(f"Registros entrenamiento: {len(X_train)}")
print(f"Registros prueba: {len(X_test)}")
print(f"Pendiente: {modelo.coef_[0]:.4f} millones toneladas/año")
print(f"Intercepto: {modelo.intercept_:.2f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² prueba: {r2:.3f}")

# Tabla real vs predicho
resultados = pd.DataFrame({
    "Año": X_test["year"].values,
    "Emisiones reales": y_test.values,
    "Emisiones estimadas": predicciones
})
display(resultados)

plt.figure(figsize=(11,5))
plt.plot(X_train["year"], y_train, label="Entrenamiento", marker="o")
plt.plot(X_test["year"], y_test, label="Valores reales prueba", marker="o")
plt.plot(X_test["year"], predicciones, label="Predicción lineal", marker="x", linestyle="--")
plt.title("Regresión lineal: Emisiones CO₂ Colombia")
plt.xlabel("Año")
plt.ylabel("Millones toneladas CO₂")
plt.legend()
plt.tight_layout()
plt.show()

# Interpretación pendiente
direccion = "aumentaron" if modelo.coef_[0]>0 else "disminuyeron"
print(f"Las emisiones {direccion} en promedio {abs(modelo.coef_[0]):.3f} millones toneladas/año")
print(f"MAE prueba: {mae:.3f} | RMSE prueba: {rmse:.3f}")

if r2 >= 0:
    print(f"El modelo explica el {r2*100:.1f} % de la variación")
else:
    print("R² negativo: el modelo predice peor que el promedio")


plt.figure(figsize=(11,5))
plt.plot(X_train["year"], y_train, label="Entrenamiento", marker="o")
plt.plot(X_test["year"], y_test, label="Valores reales prueba", marker="o")
plt.plot(X_test["year"], predicciones, label="Predicción lineal", marker="x", linestyle="--")
plt.title("Regresión lineal: Emisiones CO₂ Colombia")
plt.xlabel("Año")
plt.ylabel("Millones toneladas CO₂")
plt.legend()
plt.tight_layout()
plt.show()

# Interpretación pendiente
direccion = "aumentaron" if modelo.coef_[0]>0 else "disminuyeron"
print(f"Las emisiones {direccion} en promedio {abs(modelo.coef_[0]):.3f} millones toneladas/año")
print(f"MAE prueba: {mae:.3f} | RMSE prueba: {rmse:.3f}")

if r2 >= 0:
    print(f"El modelo explica el {r2*100:.1f} % de la variación")
else:
    print("R² negativo: el modelo predice peor que el promedio")

# Análisis Emisiones CO₂ Colombia
## Integrantes
Erika Cortes, Ximena Celix, Isabella Cardales
## Curso
Introducción a Ciencias de Datos
## Objetivo
Limpiar, visualizar emisiones anuales de CO₂ de Colombia y ajustar un modelo de regresión lineal (año vs emisiones)
## Fuente
Our World in Data CO₂ emissions dataset
## Archivo principal
proyecto‑co2‑colombia.ipynb
## Evidencia grupal
Los commits realizados por cada integrante demuestran la participación colaborativa