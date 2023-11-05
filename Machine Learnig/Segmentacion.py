# Databricks notebook source
# MAGIC %md
# MAGIC ## Modelo Segmentación cliente 

# COMMAND ----------

# MAGIC %md
# MAGIC #Integrantes del grupo
# MAGIC - Angela Claudia Caquegua C.
# MAGIC - Maria Cecilia Vargas
# MAGIC - Elio Duran
# MAGIC - Alvaro Montesinos
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC Un sparkSession es un punto de entrada para programar Spark con Dataframes (DataFrame API)
# MAGIC
# MAGIC El método builder es el constructor de instancias SparkSession
# MAGIC
# MAGIC https://spark.apache.org/docs/3.1.1/api/python/reference/api/pyspark.sql.SparkSession.html

# COMMAND ----------

spark = SparkSession.builder.appName("Segmentacion").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Leer de un HDFS que se encuentra en la ruta /user/curso/ el archivo csv: churn_Operacion_2.csv
# MAGIC
# MAGIC El Schema se infiere
# MAGIC
# MAGIC Para el caso de Colab, se lee de la carpeta content

# COMMAND ----------

df_telco = spark.read.csv('/FileStore/churn_Operacion_2.csv', header=True, inferSchema=True)

# COMMAND ----------

df_telco.show(1)

# COMMAND ----------

display(df_telco)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Que es el esquema?
# MAGIC
# MAGIC Es la metadata asociada al dataFrame **df_telco**

# COMMAND ----------

df_telco.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ¿Por qué es importante definir un schema?
# MAGIC
# MAGIC Es importante porque los modelos por si mismos no identifican si una variable es cuantitativa o cualitativa. Por ejemplo: si la ciudades de Cochabamba y Santa Cruz por los números 1 y 2. Estos valores son númericos? Una computadora los reconoce como numéricos si no se indentifica como texto que es lo que realmente son (cualitativos)
# MAGIC

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, DoubleType, FloatType

# COMMAND ----------

Mischema = StructType(
    [
        StructField('state', StringType(), True),
        StructField('account', StringType(), True),
        StructField('area_code', StringType(), True),
        StructField('phone_number', StringType(), True),
        StructField('international_plan', StringType(), True),
        StructField('voice_mail_plan', StringType(), True),       
        StructField("number_vmail_messages",  DoubleType(), True),
        StructField('total_day_minutes', DoubleType(), True), 
        StructField('total_day_calls', DoubleType(), True),
        StructField('total_day_charge', DoubleType(), True),
        StructField('total_eve_minutes', DoubleType(), True),
        StructField('total_eve_calls', DoubleType(), True),
        StructField('total_eve_charge', DoubleType(), True),
        StructField('total_night_minutes', DoubleType(), True),
        StructField('total_night_calls', DoubleType(), True),
        StructField('total_night_charge', DoubleType(), True),
        StructField('total_intl_minutes', DoubleType(), True),
        StructField('total_intl_calls', DoubleType(), True),
        StructField('total_intl_charge', DoubleType(), True),
        StructField('number_customer_service_calls', DoubleType(), True),
        StructField('churn.', StringType(), True)       
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Re leer el dataframe con el schema acordado

# COMMAND ----------

df_telco = spark.read.csv('/FileStore/churn_Operacion_2.csv', header=True, schema=Mischema)

# COMMAND ----------

# MAGIC %md
# MAGIC La variable "churn." tiene un punto en la parte final

# COMMAND ----------

df_telco = df_telco.withColumnRenamed('churn.','churn')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Explorar datos
# MAGIC
# MAGIC Ordenar por columnas 
# MAGIC
# MAGIC ¿Cuáles son las 5 personas que menos reclamos y las 5 personas que mas reclaman?
# MAGIC
# MAGIC Ascendente

# COMMAND ----------

df_telco.sort('number_customer_service_calls')[['number_customer_service_calls', 'phone_number']].show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Descendente

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df_telco.sort(F.desc('number_customer_service_calls'))[['number_customer_service_calls', 'phone_number']].show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC Para verificar el dominio de variables cualitativas se usan funciones de agrupación
# MAGIC
# MAGIC ¿Cuál es la distribución por area?
# MAGIC
# MAGIC ¿Cual es la distribucion de la tenencia de un plan de llamadas internacionales?
# MAGIC
# MAGIC ¿Cual es la media de mensajes de texto y llamadas al call center por estado?
# MAGIC

# COMMAND ----------

df_telco.groupBy(F.col('area_code')).count().show()

# COMMAND ----------

df_telco.groupBy(F.col('international_plan')).count().show()

# COMMAND ----------

display(df_telco.groupBy(F.col('international_plan')).count())

# COMMAND ----------

df_telco[['state','number_vmail_messages','number_customer_service_calls']].groupBy(F.col('state')).avg().show()

# COMMAND ----------

df_telco.groupBy(F.col('voice_mail_plan')).count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ¿Como puedo graficar?
# MAGIC
# MAGIC Hay varias fomas de hacerlo. Una es crer un resumen con Spark para llevarlo a Pandas y usar alguna libreria de graficación.
# MAGIC
# MAGIC Para ello se usa el metodo **toPandas**

# COMMAND ----------

df_telco_pandas = df_telco[['churn','number_customer_service_calls']].groupBy(F.col('churn')).avg().toPandas()

# COMMAND ----------

type(df_telco_pandas)

# COMMAND ----------

type(df_telco)

# COMMAND ----------

df_telco_pandas

# COMMAND ----------

df_telco_pandas.plot.bar('churn', figsize=(10,5))
plt.show()

# COMMAND ----------

df_telco_pandas = df_telco[['churn','total_day_minutes']].groupBy(F.col('churn')).avg().toPandas()

# COMMAND ----------

df_telco_pandas.plot.bar('churn', figsize=(10,5))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##   Transformaciones para refinar datos
# MAGIC
# MAGIC ###   Renonbrar columnas
# MAGIC
# MAGIC La variable churn. contiene un punto al final que podemos eliminarlo

# COMMAND ----------

# ya se hizo lineas arriba
# df_telco = df_telco.withColumnRenamed('churn.','churn')

# COMMAND ----------

df_telco.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ### Eliminación de valores nulos

# COMMAND ----------

df_telco_describe = df_telco.describe().toPandas()

# COMMAND ----------

df_telco_describe

# COMMAND ----------

df_telco_summary = df_telco.summary().toPandas()

# COMMAND ----------

df_telco_summary

# COMMAND ----------

final_df = df_telco.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transformación de registros
# MAGIC
# MAGIC En general para realizar una transformación a una columna se usa la función withColumn
# MAGIC
# MAGIC * Borrar prosibles espacios vacios. Se usa la funcion **trim**

# COMMAND ----------

final_df.groupBy(F.col('churn')).count().show()

# COMMAND ----------

final_df = final_df.withColumn('churn', F.trim(final_df.churn))

# COMMAND ----------

final_df.groupBy(F.col('churn')).count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering
# MAGIC
# MAGIC Se realiza una transformación de variables cualitativas.
# MAGIC
# MAGIC Los modelos de ML no interpretan palabras o textos, estos deben ser convertidos a números.
# MAGIC
# MAGIC Una de estas transformaciones es el one-hot-encoding
# MAGIC
# MAGIC ![image.png](data:image/png;base64,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)

# COMMAND ----------

df_telco_final = final_df.select([
 'state',
 'area_code',
 'international_plan',
 'voice_mail_plan',
 'number_vmail_messages',
 'total_day_minutes',
 'total_day_calls',
 'total_day_charge',
 'total_eve_minutes',
 'total_eve_calls',
 'total_eve_charge',
 'total_night_minutes',
 'total_night_calls',
 'total_night_charge',
 'total_intl_minutes',
 'total_intl_calls',
 'total_intl_charge',
 'number_customer_service_calls',
 "churn"])

# COMMAND ----------

# MAGIC %md
# MAGIC Para realizar ingeniería de caracteristicas pyspark tienen una serie de herramientas disponibles en el modulo **feature**
# MAGIC
# MAGIC **VectorAssembler** Es un transformador que combina una lista de columnas en un único vector columna. Es usado para combinar columnas originales con transformadas en un simple vector.
# MAGIC
# MAGIC **OneHotEncoder** Realiza la transformación one-hot-encoding
# MAGIC
# MAGIC **StringIndexer** Codifica una columna de etiquetas a una columna de etiquetas de números. El orden de asignación de etiquetas numéricas esta en función a la frecuencia de las etiquetas de texto. Significa que 0 corrresponde a la etiqueta mas frecuente y asi.
# MAGIC
# MAGIC
# MAGIC https://spark.apache.org/docs/latest/ml-features
# MAGIC

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer

# COMMAND ----------

# MAGIC %md
# MAGIC one hot encoding international_plan

# COMMAND ----------

international_plan_indexer = StringIndexer(inputCol='international_plan', outputCol= 'international_planIndex')

# COMMAND ----------

type(international_plan_indexer)

# COMMAND ----------

international_plan_encoder = OneHotEncoder(inputCol = 'international_planIndex', outputCol= 'international_planVec')

# COMMAND ----------

type(international_plan_encoder)

# COMMAND ----------

# MAGIC %md
# MAGIC one hot encoding State

# COMMAND ----------

state_indexer = StringIndexer(inputCol= 'state', outputCol= 'stateIndex')

# COMMAND ----------

state_encoder = OneHotEncoder(inputCol = 'stateIndex', outputCol= 'stateVec')

# COMMAND ----------

# MAGIC %md
# MAGIC one hot encoding area_code

# COMMAND ----------

area_code_indexer = StringIndexer(inputCol='area_code', outputCol= 'area_codeIndex')

# COMMAND ----------

area_code_encoder = OneHotEncoder(inputCol = 'area_codeIndex', outputCol= 'area_codeVec')

# COMMAND ----------

# MAGIC %md
# MAGIC one hot encoding voice_mail_plan

# COMMAND ----------

voice_mail_plan_indexer = StringIndexer(inputCol='voice_mail_plan', outputCol= 'voice_mail_planIndex')

# COMMAND ----------

voice_mail_plan_encoder = OneHotEncoder(inputCol = 'voice_mail_planIndex', outputCol= 'voice_mail_planVec')

# COMMAND ----------

churn_indexer = StringIndexer(inputCol= 'churn', outputCol= 'churnIndex')

# COMMAND ----------

# 'stateVec',
assembler = VectorAssembler(inputCols = [ 'area_codeVec', 'international_planVec', 'voice_mail_planVec',
                                        'number_vmail_messages',
                                         'total_day_minutes',
                                         'total_day_calls',
                                         'total_day_charge',
                                         'total_eve_minutes',
                                         'total_eve_calls',
                                         'total_eve_charge',
                                         'total_night_minutes',
                                         'total_night_calls',
                                         'total_night_charge',
                                         'total_intl_minutes',
                                         'total_intl_calls',
                                         'total_intl_charge',
                                         'number_customer_service_calls'
                                        ], outputCol= 'features')

# COMMAND ----------

# MAGIC %md
# MAGIC Seleccionando las variables que se usarán en el análisis cluster

# COMMAND ----------

df_telco_final.columns

# COMMAND ----------

#  Quitando la variable churn que sirve para un modelo de aprendizaje supervizado

df_telco_col_seg = df_telco_final.select([
 # 'state',
 'area_code',
 'international_plan',
 'voice_mail_plan',   
 'number_vmail_messages',
 'total_day_minutes',
 'total_day_calls',
 'total_day_charge',
 'total_eve_minutes',
 'total_eve_calls',
 'total_eve_charge',
 'total_night_minutes',
 'total_night_calls',
 'total_night_charge',
 'total_intl_minutes',
 'total_intl_calls',
 'total_intl_charge',
 'number_customer_service_calls'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modelado 
# MAGIC
# MAGIC Debido a que el problema de negocio tiene que ver con la identificación de grupos, se usará un algoritmo de clustering que se basa en distancias con la finalidad de crear grupos que se convertiran en segmentos.
# MAGIC
# MAGIC ### Pipelines
# MAGIC
# MAGIC Los ML Pipelines proveen un conjunto de APIS construidas sobre los DataFrames que ayudan a crear y tunear pipelines de ML.
# MAGIC
# MAGIC Se pueden combinar multiples algoritmos en un simple pipeline.
# MAGIC
# MAGIC - **DataFrame** Esta ML API usa dataframes Spark SQL, los cuales al igual que Pandas pueden almacenar diferentes tipos de datos. Ej: Un dataframe puede almacenar texto, vectores de caracteristicas, etiquetas verdadero/falso y predicciones.
# MAGIC
# MAGIC - **Transformer**  Un transformer es un algoritmo que puede transformar un dataframe en otro dataframe. Ej: un modelo de ML es un transformer que transforma un dataframa con caracteristicas en un dataframe con predicciones.
# MAGIC
# MAGIC - **Estimador** Un estimador es un algoritmo que puede ajustar sombre un dataframe para producir un transformer. Ej. Un algoritmo de aprendizaje es un estimador el cual entrega sobre un dataframe y produce un modelo
# MAGIC
# MAGIC - **Pipeline** Un pipeline es una cadena de multiples transformaciones y estimadores juntos para especificar un flujo de trabajo ML.
# MAGIC
# MAGIC ## Componentes de un pipeline
# MAGIC
# MAGIC En ML es común ejecutar una secuencia de algoritmos para procesar y aprender de los datos. Ej:
# MAGIC
# MAGIC - Realizar un one hot encoding
# MAGIC
# MAGIC MLlib representado como un flujo de Pipeline, el cual consiste de uns secuencia de PipelineStages (Estimadores y transformers) para ser ejecutados en un orden específico. 
# MAGIC
# MAGIC Para los stage transformadores, el método transform() es llamado sobre el DataFrame. Para los stage estimadores, el método fit() es llamado.
# MAGIC
# MAGIC
# MAGIC https://spark.apache.org/docs/latest/ml-pipeline.html
# MAGIC
# MAGIC https://spark.apache.org/docs/latest/ml-clustering.html
# MAGIC
# MAGIC https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.ClusteringEvaluator.html#pyspark.ml.evaluation.ClusteringEvaluator
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

from pyspark.ml import Pipeline

# COMMAND ----------

kmeans = KMeans().setK(5).setSeed(1)

# COMMAND ----------

pipeline = Pipeline(stages= [
    # state_indexer,
    area_code_indexer,
    international_plan_indexer,
    voice_mail_plan_indexer,
    # state_encoder,
    area_code_encoder,
    international_plan_encoder,
    voice_mail_plan_encoder,
    assembler, 
    kmeans])

# COMMAND ----------

train_data, test_data = df_telco_col_seg.randomSplit([0.7,0.3])

# COMMAND ----------

fit_model = pipeline.fit(train_data)

# COMMAND ----------

type(fit_model)

# COMMAND ----------

# MAGIC %md
# MAGIC realizar clusterizaciones

# COMMAND ----------

predictions = fit_model.transform(test_data)

# COMMAND ----------

predictions.show(10)

# COMMAND ----------

predictions.sort(F.desc('voice_mail_plan')).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC para observar de mejor manera se puede obtener una muestra y convertirlo en un dataframe Pandas

# COMMAND ----------

predictions_pandas = predictions.sample(fraction=0.5).toPandas()

# COMMAND ----------

predictions_pandas.head()

# COMMAND ----------

predictions.groupBy(F.col('prediction')).count().show()

# COMMAND ----------

predictions_pandas['prediction'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluar mediante el coeficiente de Silhouette
# MAGIC
# MAGIC El coeficiente Silhouette se encuentra entre - 1 y 1 siendo lo mejor 1 y lo peor -1, el valor 0 indica que los clusters estan sobreponiendo

# COMMAND ----------

evaluador = ClusteringEvaluator()

# COMMAND ----------

silhouette = evaluador.evaluate(predictions)
print("El coeficiente Silhouette usando distancias euclidianas al cuadrado es = " + str(silhouette))
