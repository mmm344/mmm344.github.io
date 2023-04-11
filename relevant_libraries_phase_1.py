# Databricks notebook source
pip install geopandas

# COMMAND ----------

pip install geopy

# COMMAND ----------

pip install pycountry_convert

# COMMAND ----------

import re

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, FloatType
from pyspark.sql.functions import mean, min, max, count, regexp_replace, percentile_approx, col, when, isnan, sum
from pyspark.ml.feature import OneHotEncoder, StringIndexer


from pycountry_convert import country_name_to_country_alpha2, country_alpha2_to_continent_code
import geopy

from delta.tables import *
