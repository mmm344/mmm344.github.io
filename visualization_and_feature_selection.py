# Databricks notebook source
# MAGIC %md
# MAGIC # Phase II: Visualization and Feature Selection
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC In this phase, data is analyzed and visualized to identify patterns and insights.

# COMMAND ----------

# MAGIC %run ./relevant_libraries_phase_2

# COMMAND ----------

# MAGIC %run ./utilities 

# COMMAND ----------

# MAGIC %md
# MAGIC Let's first read our delta file from Phase I

# COMMAND ----------

phase_1_path = 'dbfs:/user/hive/warehouse/preprocessing'
df = spark.read.format("delta").load(phase_1_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #Descriptive Statistics

# COMMAND ----------

# MAGIC %md
# MAGIC Let's grab some descriptive statistics for scores.

# COMMAND ----------


df.select('happiness_score').describe().show()

# COMMAND ----------

df.select(percentile_approx("happiness_score", 0.25).alias("25th_percentile"),
                                percentile_approx("happiness_score", 0.5).alias("50th_percentile"),
                                percentile_approx("happiness_score", 0.75).alias("75th_percentile")).show()

# COMMAND ----------

def percentile_val(df,col,perc):
    df = df.select(percentile_approx(col,perc).alias("percentile_val"))
    return df

# COMMAND ----------

(percentile_val(df, 'happiness_score', 0.5)).show()

# COMMAND ----------


# happiness_df = df.select('country','happiness_score').where(col('happiness_score') > 5.559).toPandas()

grouped_df = df.groupBy('country').agg(max("happiness_score").alias("desc_happiness_score"))
top_10_df = grouped_df.sort('desc_happiness_score', ascending=False).limit(10)
top_10_df = top_10_df.toPandas()
display(top_10_df)


# COMMAND ----------

# fig, ax = plt.subplots(figsize=(15, 6))
# ax.bar(top_10_df['country'], top_10_df['desc_score'], color='b')
# ax.set_title('Top 10 Countries by Happiness Score')
# ax.set_xlabel('Country')
# ax.set_ylabel('Happiness Score')
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Visualization
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC Let's get a correlation heatmap as a part of our investigation

# COMMAND ----------

num_df = df.select(df.columns[2:9])

# COMMAND ----------

# MAGIC %md
# MAGIC Convert to pandas dataframe

# COMMAND ----------

num_df=num_df.toPandas()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(15, 6))
sns.heatmap(num_df.corr())

# COMMAND ----------

# MAGIC %md
# MAGIC Where do we see correlation above 0.7? Let's find them and list the vertically as ordered pairs for easy viewing. 

# COMMAND ----------

corr_df = num_df.corr()
high_corr_pairs = []
for i in range(len(corr_df.columns)):
    for j in range(i+1, len(corr_df.columns)):
        if corr_df.iloc[i, j] > 0.7:
            high_corr_pairs.append((corr_df.columns[i], corr_df.columns[j]))

high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x[1], reverse=True)


# print the high correlation pairs and their scores vertically
for pair in high_corr_pairs:
    col1, col2 = pair
    corr_score = corr_df.loc[col1, col2]
    print("{} and {} with score: {}".format(col1, col2, corr_score))



# COMMAND ----------

# MAGIC %md
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC Specifically, let's investigate ``country`` and ``happiness_score``.

# COMMAND ----------

test_df = df.select('country','happiness_score').toPandas()

# COMMAND ----------

# Load the world map shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a list of countries
# countries = ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina']

# Generate artificial happiness scores for each country
# scores = pd.DataFrame({'Country': countries, 'Happiness Score': np.random.rand(len(countries))})

scores = test_df = df.select('country','happiness_score').toPandas()

# Merge the world map with the happiness scores data
world = world.merge(scores, left_on='name', right_on='country')

# Define the color map for the happiness scores
cmap = 'Reds'

# Plot the map with happiness scores as colors
fig, ax = plt.subplots(figsize=(15,25))
ax.set_aspect('equal')
world.plot(
    ax=ax,
    column='happiness_score',
    cmap=cmap
)
# Create a separate axis for the colorbar
cax = fig.add_axes([1, 0.35, 0.05, 0.3])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=world['happiness_score'].min(), vmax=world['happiness_score'].max()))
sm._A = []
fig.colorbar(sm, cax=cax)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next, let's create scatter plots between the different variables to visually inspect linear relationships, if they exist at all.

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

pd.plotting.scatter_matrix(num_df, diagonal="kde",figsize=(20,15))
plt.show()

# COMMAND ----------

num_sdf = spark.createDataFrame(num_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Do the same thing but using Databricks' visualization tools.

# COMMAND ----------

display(num_sdf.select('explained_by_gdp_per_capita', 'explained_by_healthy_life_expectancy'))

# COMMAND ----------

# MAGIC %md
# MAGIC # PCA/MI Investigation 

# COMMAND ----------

# MAGIC %md
# MAGIC Principle componenet analysis: The main idea is that instead of describing the data with the original features, we describe it with its axes of variation - thus reducing the overall dimensionality. This is sometimes handled in the preprocessing phase, but it has visual implications as well, so we show it here. 
# MAGIC 
# MAGIC Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.
# MAGIC 
# MAGIC By examining the percentage of variance explained by each principal component, we can determine the number of principal components needed to retain a desired amount of variance in the data. This can help us to reduce the dimensionality of the dataset while still retaining most of the important information or variability in the data.

# COMMAND ----------

# MAGIC %md
# MAGIC We can perform a quick check to see if PCA or t-SNE is more appropriate. 

# COMMAND ----------

rec = pca_or_tsne(df)

print('Here is the recommended method:', rec)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's get the shape of our data

# COMMAND ----------

num_df.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's split ``num_df`` in terms of features and target, ``X`` and ``y``, respectively. From there, we will create a PCA instance and create the princpal components for ``num_df``. After inspection, we see that 3 principal components is enough for this dataset (representing roughly 80% of the variance). 

# COMMAND ----------

features = ['explained_by_gdp_per_capita', 'explained_by_social_support',
       'explained_by_healthy_life_expectancy',
       'explained_by_freedom_to_make_life_choices', 'explained_by_generosity',
       'explained_by_perceptions_of_corruption']
X = num_df.copy()
y = num_df.pop('happiness_score')
X = X.loc[:, features]
#create a PCA instance
n_components = 3
pca = PCA(n_components = n_components) 

#create princpal components for num_df
X_pca = pca.fit_transform(num_df)

# create headers and convert to dataframe
component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
X_pca = pd.DataFrame(X_pca, columns=component_names)

X_pca.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The PCA instance loads the transformed components of X into its ``componenets_`` attribute. Let's display those. 

# COMMAND ----------

X.columns

# COMMAND ----------

loadings = pd.DataFrame(
    pca.components_.T,  # transpose the matrix of loadings
    columns=component_names,  # so the columns are the principal components
    index=X.columns,  # and the rows are the original features
)
loadings

# COMMAND ----------

loadings.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## A little about Loadings
# MAGIC Each element in the ``loadings`` DataFrame represents the correlation between an original feature and a principal component. The larger the absolute value of the element, the stronger the correlation between the feature and the component. If an element is the sign determines the type of correlation (positive or negative). 

# COMMAND ----------

# MAGIC %md
# MAGIC Let's plot the variance from pca

# COMMAND ----------

plot_variance(pca)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that this plot shows how our features vary along each axis. The first axis has the most percent explained variance, and the first two axes together describe roughly 75% of the overall variation. 

# COMMAND ----------

# MAGIC %md
# MAGIC The PCA instance also loads ``explained_varaince_ratio``, which provides the proportion of variance in the original data that is explained by each principal component.

# COMMAND ----------

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# COMMAND ----------

# MAGIC %md
# MAGIC From here we see that the three components make up ~80% of the information. So, if we were to project 145 samples along 3 axes, we lose ~20% of the information from the dataset. 

# COMMAND ----------

# MAGIC %md
# MAGIC By plotting the principal components against one another, we can visualize how different variables relate to each other and identify patterns in the data. For example, if two principal components are plotted against each other and form a distinct cluster, this may indicate that the variables they represent are strongly correlated with each other. Alternatively, if the points on the plot are scattered randomly, this may suggest that there is no strong correlation between the variables.

# COMMAND ----------

pd.plotting.scatter_matrix(loadings, diagonal="kde",figsize=(20,15))
plt.show()

# COMMAND ----------

pd.plotting.scatter_matrix(X_pca, diagonal="kde",figsize=(20,15))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The mutual information (MI) score is a measure of the dependence between two variables. In the context of this function, the mutual information score between a feature and the target variable is a measure of how much information the feature provides about the target variable. A high mutual information score indicates that the feature is highly informative about the target variable, while a low score indicates that the feature provides little information about the target variable.

# COMMAND ----------

mi_scores = make_mi_scores(X_pca, y, discrete_features=False)
display(mi_scores)

# COMMAND ----------

# MAGIC %md
# MAGIC #Feature Selection
# MAGIC ____________________________________________________________________________________________
# MAGIC 
# MAGIC There are different ways in which we can perform feature selection. Here, we use sklearn's ``feature_selection`` module help with this task. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA/MI Method
# MAGIC _______________________________________________________________________________________________

# COMMAND ----------

# MAGIC %md
# MAGIC Let's combine PCA and MI scores for feature selection. ``X_selected_vx`` represents the most important features that contribute to the variance in the data or most strongly associated with ``happiness_score``. This step helps with overfitting and performance for modeling later. 

# COMMAND ----------

X = df.drop('happiness_score', axis = 1).drop('country', axis =1) #we encoded country already
# Define a pipeline to combine PCA and mutual information regression scores
pca_mi_pipeline = Pipeline([
    ('pca', PCA(n_components=3)),
    ('mi', SelectKBest(score_func=mutual_info_regression, k=3))
])

#encode country 
le = LabelEncoder()
# X['country'] = le.fit_transform(X['country'])

# Fit the pipeline to the data and transform the features
X_selected_v1 = pca_mi_pipeline.fit_transform(X, y)


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ###The variance threshold method
# MAGIC _______________________________________________________________________________________________
# MAGIC The purpose of this method is to remove the features that have low variance in the dataset. Notice there we grab all the features and instead of a select few as above. Note the difference in the encoding step - here we are keeping the country column but replacing the string values with the encoding. It is an alternative to the method used previously. 

# COMMAND ----------

X = df.drop('happiness_score', axis = 1)

#encode country 
le = LabelEncoder()
X['country'] = le.fit_transform(X['country'])
# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

y = df['happiness_score']

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_selected_v2 = sel.fit_transform(X)

# COMMAND ----------

# MAGIC %md
# MAGIC Save ``X_selected_v1``, ``X_selected_v2``, and ``y`` for the modeling phase.

# COMMAND ----------

np.save('/dbfs/user/hive/warehouse/X_selected_v1.npy', X_selected_v1)
np.save('/dbfs/user/hive/warehouse/X_selected_v2.npy', X_selected_v1)
np.save('/dbfs/user/hive/warehouse/y.npy', y)