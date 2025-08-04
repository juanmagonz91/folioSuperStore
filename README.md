# folioSuperStore

Análisis de Ventas y Rentabilidad de Super Store

Este repositorio contiene el código y el análisis exploratorio y predictivo de un dataset de ventas de una Super Store, enfocado en identificar factores que impactan la rentabilidad de las transacciones y visualizar los hallazgos en un dashboard interactivo.

Contenido del Repositorio

app.py: Script principal de la aplicación Streamlit que genera el dashboard interactivo.
requirements.txt: Archivo que lista las dependencias de Python necesarias para ejecutar la aplicación.
README.md: Este archivo.

Análisis Realizado
El proyecto llevó a cabo un análisis en dos fases:
Análisis Exploratorio de Datos (EDA):
Carga y limpieza del dataset.
Análisis de dimensiones y tipos de datos.
Identificación y manejo de duplicados.
Análisis de distribuciones de variables numéricas (Ventas, Cantidad, Descuento, Ganancia) usando histogramas y boxplots.
Análisis de frecuencia de variables categóricas (Modo de Envío, Segmento, Región, Categoría, Subcategoría).
Análisis de rentabilidad agregada por Categoría y Subcategoría, identificando áreas problemáticas (especialmente "Bookcases" y "Tables" en "Furniture").
Investigación de la relación entre Descuento y Ganancia para las subcategorías con pérdidas.

Análisis Predictivo:
Definición del objetivo: predecir la probabilidad de que una transacción sea no rentable.
Preparación de datos para modelado (creación de variable objetivo, codificación one-hot).
Entrenamiento de un modelo Gradient Boosting Classifier.
Evaluación del modelo utilizando métricas de clasificación (Exactitud, Precisión, Recall, F1-score, AUC).
Análisis de importancia de características, confirmando el descuento como el factor más influyente en la falta de rentabilidad.
Ajuste del umbral de clasificación para optimizar la capacidad del modelo para identificar transacciones no rentables (mejorar el Recall).

Dashboard Interactivo (Streamlit)
El archivo app.py contiene el código para un dashboard interactivo construido con Streamlit. Este dashboard permite a los usuarios:
Explorar distribuciones de datos y tablas resumen filtradas por Región, Subcategoría y Rango de Descuento.
Visualizar la relación entre Descuento y Ganancia con filtros aplicados.
Utilizar el modelo predictivo entrenado para predecir la rentabilidad de una transacción hipotética basada en los detalles ingresados.
Ver la importancia de las características del modelo y sus métricas de evaluación.
