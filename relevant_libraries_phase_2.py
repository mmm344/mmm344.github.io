# Databricks notebook source
pip install geopandas

# COMMAND ----------

pip install geopy

# COMMAND ----------

pip install pycountry_convert

# COMMAND ----------

pip install h3

# COMMAND ----------

import re

import pandas as pd

from pyspark.sql.functions import lower, col, percentile_approx, max

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from delta import DeltaTable
from delta.tables import *

import matplotlib.pyplot as plt

import seaborn as sns

import geopandas as gpd

