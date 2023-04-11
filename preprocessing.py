# Databricks notebook source
# MAGIC %md
# MAGIC # Phase I: Preprocessing/Data Cleansing
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC In this phase we import the raw data and perform essential preprocessing of our World Happiness dataset. The final dataframe will be written to a delta table called ``preprocessing``.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Import necessary libraries for phase I
# MAGIC 2. Import utility notebook
# MAGIC 3. Create a spark session called ``spark``

# COMMAND ----------

# MAGIC %run ./relevant_libraries_phase_1

# COMMAND ----------

# MAGIC %run ./utilities 

# COMMAND ----------

spark = SparkSession.builder.appName("World Happiness Report").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Import the raw data

# COMMAND ----------

df = spark.read.format("hive").option("header", "true").option("inferSchema", "true").table("year_2022")


# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC Check the data type to see if any adjustments need to be made.

# COMMAND ----------

#Let's take a look at the columns and their types
df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that we need to convert some string types to float types for machine learning purposes. Let's first rename our columns so they're easier to work with and we don't have to worry about special characters.

# COMMAND ----------

df = clean_column_names(df)
display(df)

# COMMAND ----------

df = (df.withColumnRenamed("Explained by: Generosity", "generosity")
      .withColumnRenamed("Explained by: Perceptions of corruption", "perception_of_corruption")
      .withColumn('country', when(col('country') == 'Taiwan Province of China', 'Taiwan').otherwise(col('country')))
      .withColumn('country', when(col('country') == 'Hong Kong S.A.R. of China', 'Hong Kong').otherwise(col('country'))))

# COMMAND ----------

df.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC For this dataset, let's drop ``whisker_high``, ``whisker_low``, and ``dystopia_183_residual``

# COMMAND ----------

df = df.drop('whisker_high')
df = df.drop('whisker_low')
df = df.drop('dystopia_183_residual')

# COMMAND ----------

# MAGIC %md
# MAGIC Let's remove the commas and cast as ``DoubleType()``

# COMMAND ----------

df = clean_dataframe(df)

# COMMAND ----------

numeric_type = DoubleType()
for col_name in df.columns[2:]:
    df = df.withColumn(col_name, col(col_name).cast(numeric_type))

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC Investigate NA values relative to ``happiness_score``

# COMMAND ----------

null_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])

display(null_counts)

# Assuming 'df' is the name of your DataFrame
null_cols = df.columns[2:] # List of column names

# conditions = [(col(c).isNull()) for c in null_cols]

conditions = [when(col(c).isNull(), 1).otherwise(0).alias(c+'_null') for c in null_cols]

# Filter the rows where happiness_score is null and group by country
null_scores_by_country = (df.select("*", *conditions).filter(" OR ".join([c+'_null=1' for c in null_cols]))
.groupBy("country")
.count())

# Display the countries with null happiness scores and the number of occurrences
display(null_scores_by_country)


# COMMAND ----------

# MAGIC %md
# MAGIC Drop NA values

# COMMAND ----------

# Drop all rows with any null values
df = df.na.drop()

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC Let's see if we have any duplicate rows

# COMMAND ----------

duplicate_row = df.groupBy(df.columns).count().where(col("count") > 1)
display(duplicate_row)

# COMMAND ----------

# MAGIC %md
# MAGIC We can drop the duplicates, but we don't see any for this data set. 

# COMMAND ----------

df

# COMMAND ----------

df = df.dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's utilize preprocessing tools from Pandas to perform one-hot encoding and normalization using ``MinMaxScaler``. 

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC Let's encode the categorical variables. We will use one-hot encoding on ``df_pd``.

# COMMAND ----------

df = encode_and_bind(df, 'country')

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC Next, we will standardize ``pd_df``. Recall that df is a Spark ``pyspark.sql.dataframe.DataFrame``. We will do this in the following steps:
# MAGIC 1. Select the columns we want to normalize 
# MAGIC 2. Assemble the selected columns into a vector column
# MAGIC 3. Scale and normalize the vector column
# MAGIC 4. Drop the original columns and keep only the normalized features

# COMMAND ----------

# select only the numerical columns to be normalized
num_cols = df.columns[2:]
df_num = df[num_cols]

# normalize the numerical columns using MinMaxScaler
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_num), columns=num_cols)

# combine the normalized columns with the non-normalized columns
df = pd.concat([df[['rank','country']], df_normalized], axis=1)

# #convert back to pyspark dataframe
df = spark.createDataFrame(df)

df = clean_column_names(df)

# COMMAND ----------

df.select('happiness_score','country').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC Let's finishing the preprocessing step by saving out dataframe as a delta table

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists preprocessing

# COMMAND ----------

type(df)

# COMMAND ----------


# specify the Delta table path
delta_table_path = "dbfs:/user/hive/warehouse/preprocessing"

df.write.format("delta").mode("overwrite").save(delta_table_path)



# COMMAND ----------

# print(dbutils.fs.ls("/"))
