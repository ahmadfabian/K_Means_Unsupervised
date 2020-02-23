# import libararies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# instantiate numpy array
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
print(X)

#
Y = pd.DataFrame(X, columns=['A', 'B'])
print(Y)

#
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

#
kmeans.cluster_centers_

#
Y['cluster id'] = kmeans.labels_
print(Y)

# inertia
kmeans.inertia_

# kmeans prediction
kmeans.predict([[0,0], [4,4]])
