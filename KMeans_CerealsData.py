# import libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering

# import data
cereals = pd.read_csv('input/Cereals.csv')
cereals.head(10)

# data shape
cereals.shape

# data describe
cereals.describe()

cereals.describe(include='object')

# Aggregate the columns ‘name’ shelf’ and ‘rating’ into one single column and name is as ‘label’
cereals['label'] = cereals['name']+ ' (' + cereals['shelf'].astype(str) + " - " + round(cereals['rating'],2).astype(str) + ')'
cereals.drop(['name','shelf','rating'], axis=1, inplace=True)
cereals.head()

# Observe the datatypes of the data using dtypes attribute.
cereals.dtypes

# Check whether the newly created label field is unique across along the dataframe


#
cereal_label = cereals['label']
cereals.drop('label', axis=1, inplace=True)

#
imputer = SimpleImputer(strategy='mean')
imputer.fit(cereals)
imputed_cereals = imputer.transform(cereals)

#
type(imputed_cereals)

#
imputed_cerals = pd.DataFrame(imputer.transform(cereals), columns=cereals.columns)

#
imputed_cerals.head(1)

#
scaler = StandardScaler()
scaler.fit(imputed_cerals)
scaled_cereals = scaler.transform(imputed_cerals)

#
scaled_cereals = pd.DataFrame(scaler.transform(imputed_cerals), columns=cereals.columns)

#
scaled_cereals.head(1)

#
kmeans_cereals = KMeans(n_clusters=5, random_state=7)
kmeans_cereals.fit(scaled_cereals)

#
kmeans_cereals.cluster_centers_

#
kmeans_cereals.labels_

#
kmeans_cereals.inertia_

#
pd.options.display.html.table_schema = True
wss= {}
for k in range(2, 21):
    kmeans_loop = KMeans(n_clusters=k,n_init=30,n_jobs=-1,random_state=1000,verbose=0).fit(scaled_cereals)
    clusters = kmeans_loop.labels_
    labels = kmeans_loop.predict(scaled_cereals)
    print('silhouette_score(scaled_cereals, labels):', silhouette_score(scaled_cereals, labels))
    wss[k] = kmeans_loop.inertia_

#
plt.figure(figsize=(15,8))
plt.plot(list(wss.keys()),list(wss.values()) ,marker='o')
plt.grid()
plt.xlabel('Number of clusters')
plt.ylabel('Total within sum of squares')
plt.show()

#
best_kmeans = KMeans(n_clusters=9, random_state=1240)
best_kmeans.fit(scaled_cereals)
best_kmeans_labels = best_kmeans.predict(scaled_cereals)

#
kmeans_results = pd.DataFrame({"label":cereal_label,"kmeans_cluster":best_kmeans_labels})
kmeans_results.head()

#
cereals = pd.read_csv("input/Cereals.csv")
cereals['label'] = cereals['name']+ ' (' +cereals['shelf'].astype(str) + " - " + round(cereals['rating'],2).astype(str) + ')'
cereals.drop(['name','shelf','rating'], axis=1, inplace=True)
final_cluster_data = pd.merge(cereals, kmeans_results, on='label')
final_cluster_data.head(10)

#
final_cluster_data.kmeans_cluster.value_counts()

#
analysis = final_cluster_data.groupby(['kmeans_cluster']).mean().reset_index()
print(analysis)

#
linkage_matrix = linkage(scaled_cereals, method='ward',metric='euclidean')

#
dendrogram(linkage_matrix,labels=cereal_label.values)

#
agg_clust = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
agg_clusters = agg_clust.fit_predict(scaled_cereals)
agg_result = pd.DataFrame({"label":cereal_label,"agg_cluster":agg_clusters}).sort_values('agg_cluster')
agg_result.head()

#
