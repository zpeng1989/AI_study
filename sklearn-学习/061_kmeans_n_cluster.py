from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)

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

