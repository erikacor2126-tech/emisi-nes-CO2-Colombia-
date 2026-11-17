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