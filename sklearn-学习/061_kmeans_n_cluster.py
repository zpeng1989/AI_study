from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

n_clusters = 4
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
y_pred = cluster.labels_

silhouette_score(X, y_pred)
silhouette_score(X, cluster_.labels_)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

n_clusters = 4
fig, (ax1, ax2 ) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([-0.1, 1])



ax1.set_ylim([0, x.shape[0] + (n_clusters + 1) * 10)
clusterer = KMeans(n_clusters = n_clusters, random_state = 10).fit(X)
cluster_labels = clusterer.labels_

silhouette_avg = silhouette_score(X,cluster_labels)
print(n_clusters, silhouette_avg)

sample_silhouette_value = silhouette_samples(X, cluster_labels)

y_lower = 10

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_value[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i)/n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),ith_cluster_silhouette_values,facecolor = color, alpha = 0.7)
    ax1.test(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

ax1.set_title('silhouette')
ax1.set_xlabel('the silhouette coefficient')
ax1.set_ylabel('cluster label')
ax1.axvline(x = silhouette_avg, color = 'red', linestyle = '--')

centers = cluster.cluster_centers_
ax2.scatter(centers[:,0], centers[:,1], marker = 'x', c= 'red')


