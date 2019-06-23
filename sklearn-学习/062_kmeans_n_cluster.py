from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

for n_clusters in [2,3,4,5,6,7]:
    n_clusters = n_clusters
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18,7)
    ax1.set_xlim([-0.1,1])
    ax1.set_ylim([0, X.shape[0] + (n_clusters + 1)* 10])
    clusterer = KMeans(n_clusters = n_clusters, random_state = 10).fit(X)
    cluster_labels = clusterer.labels_
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(n_clusters, silhouette_avg)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i)/n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), ith_cluster_silhouette_values,facecolor = color, alpha = 0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title('the silhouette plot for the various clusters')
    ax1.set_xlabel('value')
    ax1.set_ylabel('label')
    ax1.axvline(x = silhouette_avg, color = 'red', linestyle = '--')
    colors = cm.nipy_spectral(cluster_labels.astype(float)/ n_clusters)
    ax2.scatter(X[:,0], X[:,1], marker = 'o', s = 8, c = colors)
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:,0], centers[:,1], marker = 'x', c = 'red', alpha = 1, s =200)
    plt.show()
