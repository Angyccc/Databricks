-- Databricks notebook source
-- MAGIC %md
-- MAGIC ## Analisis de Churn de la empresa TELCO

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Obteniendo los datos ingestados desde el DBMS y creandolos en una tabla

-- COMMAND ----------

DROP TABLE IF EXISTS TELCO_OPERACION1;

CREATE TABLE TELCO_OPERACION1
USING csv
OPTIONS (path "/FileStore/churn_Operacion_1.csv", header "true")


-- COMMAND ----------

describe telco_operacion1

-- COMMAND ----------

-- crear un subconjunto de la base 1
create table aux_telco_operacion1F
(customerID  String,
 gender String,
 MontlyCharges float
)

-- COMMAND ----------

select *
from aux_telco_operacion1F

-- COMMAND ----------

-- El dato string que es cuantitativo realizó un cast automatico para guardarse con el tipo correcto
describe aux_telco_operacion1F

-- COMMAND ----------

-- insertando un subconjunto de los valores a analizar
insert into aux_telco_operacion1F
select customerID, gender, MonthlyCharges
from telco_operacion1

-- COMMAND ----------

-- Crear un histograma de la variable MonthlyCharges
select MontlyCharges
from aux_telco_operacion1F

-- COMMAND ----------

DROP TABLE IF EXISTS TELCO_OPERACION2;

CREATE TABLE TELCO_OPERACION2
(
 state String,
 account String,
 area_code String,
 phone_number String,
 international_plan String,
 voice_mail_plan String,
 number_vmail_messages Float,
 total_day_minutes Float,
 total_day_calls Float,
 total_day_charge Float,
 total_eve_minutes Float,
 total_eve_calls Float,
 total_eve_charge Float,
 total_night_minutes Float,
 total_night_calls Float,
 total_night_charge Float,
 total_intl_minutes Float,
 total_intl_calls Float,
 total_intl_charge Float,
 number_customer_service_calls Float,
 churn String
 )
USING csv
OPTIONS (path "/FileStore/churn_Operacion_2.csv", header "true")


-- COMMAND ----------

select *
from telco_operacion2

-- COMMAND ----------

describe telco_operacion2

-- COMMAND ----------

-- Cual es la distribución de genero?

select gender, count(1)
from telco_operacion1
group by gender

-- COMMAND ----------

-- Cual es la distribución de personas de la tercera edad?

select SeniorCitizen, count(1) as conteo
from telco_operacion1
group by SeniorCitizen

-- COMMAND ----------

-- Cual es la distribución de churn por estado?  -- Obtenga un grafico

select state, Churn, count(*)
from telco_operacion2
group by state, Churn


-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Como llevar a un pandas?
-- MAGIC
-- MAGIC telco1_pd = spark.read.csv("/FileStore/churn_Operacion_1.csv", header="true", inferSchema="true").toPandas()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC type(telco1_pd)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC telco1_pd.head()

-- COMMAND ----------

-- MAGIC %python
-- MAGIC telco1_pd['gender'].value_counts()
