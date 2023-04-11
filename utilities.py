# Databricks notebook source
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    return(res)

# COMMAND ----------

def clean_dataframe(df):
    for column in df.columns:
        # df = df.withColumn(column, lower(col(column)))
        # df = df.withColumn(column, regexp_replace(column, " ", "_"))
        df = df.withColumn(column, regexp_replace(column, ",", ""))
        df = df.withColumn(column, regexp_replace(column, "[*]", ""))
    return df

# COMMAND ----------

def clean_column_names(df):
    """
    Cleans the column names of a PySpark SQL dataframe by making all letters lowercase,
    changing all spacing to underscores, and removing special characters like + and :.
    """
    # Define a regular expression pattern to match special characters
    pattern = r'[+:.()*]+'
    
    # Loop through the columns of the dataframe
    for column in df.columns:
        # Make all letters lowercase
        new_c = column.lower()
        
        # Change all spacing to underscores
        new_c = new_c.replace(" ", "_")

        new_c = new_c.replace("-", "_")
        
        # Remove special characters like + and :
        new_c = re.sub(pattern, '', new_c)

        #remove extra _
        new_c = new_c.replace("__", "_")
        
        # Rename the column
        df = (df.withColumnRenamed(column, new_c))
    return df

# COMMAND ----------

#this code helps plot the variance for pca; 
#reference: https://www.kaggle.com/code/ryanholbrook/principal-component-analysis#kln-21
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import mutual_info_regression


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# The function takes a trained PCA object as input and creates a figure with two subplots. The first subplot shows the percentage of variance explained by each principal component, while the second subplot shows the cumulative percentage of variance explained up to each principal component. The function returns the two subplots as an array.

def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs



# COMMAND ----------

# This code defines a function make_mi_scores that computes mutual information scores between the features and the target variable in a dataset.


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

# COMMAND ----------



def pca_or_tsne(X):
    """
    Determines whether to use PCA or t-SNE based on the size, dimensionality,
    linearity, cluster separation, and computational resources of the input data.
    
    Parameters:
    X (array-like): Input data with shape (n_samples, n_features)
    
    Returns:
    str: Either "PCA" or "t-SNE"
    """
    n_samples, n_features = X.shape
    
    # Check data size
    if n_samples > 10000:
        return "PCA"
    
    # Check dimensionality
    if n_features > 50:
        return "PCA"
    
    # Check linearity
    pca = PCA(n_components=1)
    pca.fit(X)
    if pca.explained_variance_ratio_[0] > 0.9:
        return "PCA"
    
    # Check cluster separation
    tsne = TSNE(n_components=2)
    tsne.fit(X)
    if np.min(tsne.embedding_.std(0)) < 0.05:
        return "t-SNE"
    
    # Default to PCA
    return "PCA"


