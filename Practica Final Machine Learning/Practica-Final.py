# Databricks notebook source
# MAGIC %md
# MAGIC ## Trabajo Final Aplicación Spark Machine Learning

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC **Leemos las tablas desde DBFS**

# COMMAND ----------

df_zona = spark.read.csv('/FileStore/ZONA.csv', header=True, inferSchema=True)

# COMMAND ----------

df_unidad = spark.read.csv('/FileStore/UNIDAD.csv', header=True, inferSchema=True)

# COMMAND ----------

df_tramite = spark.read.csv('/FileStore/TRAMITE.csv', header=True, inferSchema=True)

# COMMAND ----------

df_tramitecliente = spark.read.csv('/FileStore/TRAMITE_CLIENTE.csv', header=True, inferSchema=True)

# COMMAND ----------

display(df_zona)

# COMMAND ----------

display(df_unidad)

# COMMAND ----------

display(df_tramite)

# COMMAND ----------

display(df_tramitecliente)

# COMMAND ----------

# MAGIC %md
# MAGIC **Eliminar nulos**

# COMMAND ----------

df_zona_sin_nulo = df_zona.na.drop()

# COMMAND ----------

display(df_zona_sin_nulo)

# COMMAND ----------

df_unidad_sin_nulo = df_unidad.na.drop()

# COMMAND ----------

display(df_unidad_sin_nulo)

# COMMAND ----------

df_tramite_sin_nulo = df_tramite.na.drop()

# COMMAND ----------

display(df_tramite_sin_nulo)

# COMMAND ----------

df_tramitecliente_sin_nulo = df_tramitecliente.na.drop()

# COMMAND ----------

display(df_tramitecliente_sin_nulo)

# COMMAND ----------

# MAGIC %md
# MAGIC **Creamos la tabla minable**

# COMMAND ----------

df_tabla_unida = df_tramitecliente_sin_nulo.join(df_tramite_sin_nulo, "IDTRAMITE", "inner") \
              .join(df_zona_sin_nulo, "IDZONA", "inner") \
              .join(df_unidad_sin_nulo, "IDUNIDAD", "inner")

# COMMAND ----------

display(df_tabla_unida)

# COMMAND ----------

# MAGIC %md
# MAGIC **Seleccionando las variables que se usarán en el análisis cluster**

# COMMAND ----------

df_tabla_final = df_tabla_unida.select([
 'IDTRAMITE_CLIENTE',
 'IDCLIENTE',
 'TRAMITE_CLIENTENUMERO',
 'IDTRAMITE',
 'NOMBRE',
 'TRAMITE_CLIENTEESTADO',
 'ZONA_NOMBRE',
 'NOMBREUNIDAD'])

# COMMAND ----------

display(df_tabla_final)

# COMMAND ----------

df_tabla_final = df_tabla_final.withColumnRenamed("NOMBRE", "NOMBRE_TRAMITE")

# COMMAND ----------

display(df_tabla_final)

# COMMAND ----------

df_tabla_final = df_tabla_final.withColumnRenamed("NOMBREUNIDAD", "NOMBRE_UNIDAD")

# COMMAND ----------

display(df_tabla_final)

# COMMAND ----------

df_tabla_final_describe = df_tabla_final.describe().toPandas()

# COMMAND ----------

df_tabla_final_describe

# COMMAND ----------

# MAGIC %md
# MAGIC **OneHotEncoding**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder \
    .appName("Linear Regression Example") \
    .getOrCreate()

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in ["TRAMITE_CLIENTENUMERO", "NOMBRE_TRAMITE", "ZONA_NOMBRE", "NOMBRE_UNIDAD"]]
indexed = df_tabla_final
for indexer in indexers:
    indexed = indexer.fit(indexed).transform(indexed)

encoder = OneHotEncoder(inputCols=["TRAMITE_CLIENTENUMERO_index", "NOMBRE_TRAMITE_index", "ZONA_NOMBRE_index", "NOMBRE_UNIDAD_index"],
                        outputCols=["TRAMITE_CLIENTENUMERO_encoded", "NOMBRE_TRAMITE_encoded", "ZONA_NOMBRE_encoded", "NOMBRE_UNIDAD_encoded"])
encoded = encoder.fit(indexed).transform(indexed)

encoded.show(truncate=False)

# COMMAND ----------

display(encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modelado##

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC Se pretende realizar un clustering por la variable NOMBRE_UNIDAD

# COMMAND ----------

display(encoded.groupby("NOMBRE_UNIDAD").count())

# COMMAND ----------

display(encoded.groupby("IDTRAMITE").count())

# COMMAND ----------

feature_cols = ['IDTRAMITE_CLIENTE',
 'IDCLIENTE',
 'TRAMITE_CLIENTENUMERO_index',
 'IDTRAMITE',
 'NOMBRE_TRAMITE_index',
 'TRAMITE_CLIENTEESTADO',
 'ZONA_NOMBRE_index',]

vec_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data_final = vec_assembler.transform(encoded)

# COMMAND ----------

display(data_final)

# COMMAND ----------

# MAGIC %md
# MAGIC **Normalizando los datos**

# COMMAND ----------

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd =True)
scaler_model = scaler.fit(data_final)
dataScaled = scaler_model.transform(data_final)

# COMMAND ----------

display(dataScaled.select("scaled_features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Aplicando KMEANS

# COMMAND ----------

kmeans = KMeans(featuresCol="features",k=4)

# COMMAND ----------

model = kmeans.fit(dataScaled)

# COMMAND ----------

model

# COMMAND ----------

display(model.transform(dataScaled).groupBy("prediction").count())

# COMMAND ----------

predictions = model.transform(dataScaled)

# COMMAND ----------

display(predictions)

# COMMAND ----------

display(predictions.groupBy("NOMBRE_UNIDAD","prediction").count())

# COMMAND ----------


# Rango de número de clusters a probar
min_clusters = 2
max_clusters = 5
silhouette_scores = []

# Iterar sobre el rango de número de clusters
for n_clusters in range(min_clusters, max_clusters+1):
    kmeans = KMeans().setK(n_clusters).setSeed(1)  # Cambia la semilla si lo deseas
    model = kmeans.fit(dataScaled)
    predictions = model.transform(dataScaled)
    
    # Calcular el valor de silueta promedio
    evaluator = ClusteringEvaluator()
    silhouette_avg = evaluator.evaluate(predictions)
    silhouette_scores.append(silhouette_avg)

# Graficar el valor de silueta promedio para cada número de clusters
plt.plot(range(min_clusters, max_clusters+1), silhouette_scores)
plt.xlabel('Número de Clusters')
plt.ylabel('Valor de Silueta Promedio')
plt.title('Método de la Silueta')
plt.show()

print('Puntuación de la silueta:', silhouette_scores)

# COMMAND ----------

# Coeficiente de Silhouette
evaluator = ClusteringEvaluator()
silhouette_score = evaluator.evaluate(predictions)
print(f"Coeficiente de Silhouette: {silhouette_score}")
