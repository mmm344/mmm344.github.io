# Databricks notebook source
# MAGIC %md
# MAGIC # Phase III: Modeling
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC This is the final phase for this project where we make use of different models.

# COMMAND ----------

import numpy as np

# read contents of txt file into a Python list
X_selected_v1 = np.load('/dbfs/user/hive/warehouse/X_selected_v1.npy') 
X_selected_v2 = np.load('/dbfs/user/hive/warehouse/X_selected_v2.npy')
y = np.load('/dbfs/user/hive/warehouse/y.npy')

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected_v1, y, test_size=0.2, random_state=42)

# Train and evaluate the machine learning model (example using Random Forest)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
score = rf.score(X_test, y_test)
print('Model score:', score)

# COMMAND ----------

