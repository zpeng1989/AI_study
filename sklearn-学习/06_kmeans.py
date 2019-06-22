#-*- coding:utf-8 -*-
## sklearn 聚类算法
## 
## cluster.AffinityPropagation             执行亲和传播数据聚类
## cluster.AgglomerativeClustering         凝聚聚类
## cluster.Birch                           实现Birch聚类算法
## cluster.DBSCAN                          从矢量数组或者距离执行DBSCAN聚类
## cluster.FeatureAgglomeration            凝聚特征
## cluster.KMeans                          K均值聚类
## cluster.MiniBatchKMeans                 小批量K均值聚类
## cluster.MeanShift                       使用平坦核函数的凭据移位聚类
## cluster.SpecuralClustering              光谱聚类，将聚类应用于规范化拉普拉斯的投影

## param  参数
## cluster.affinity_propagation            执行亲和传播数据聚类
## cluster.dbscan                          从矢量数组或距离矩阵执行DBSCAN聚类
## cluster.estimate_bandwidth              估计使用均值平移算法的带宽
## cluster.k_means                         K均值聚类
## cluster.mean_shift                      使用平坦核聚类函数平均移位聚类
## cluster.spectral_clustering             将聚类应用于规范化拉普拉斯的投影
## cluster.ward_tree                       光谱聚类，将聚类应用于规范化拉普拉斯的投影


from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples = 500, n_features = 2, centers = 4, random_state = 1)
fig, ax1 = plt.subplots(1)
ax1.scatter(X[:, 0], X[:, 1], marker = 'o', s = 8)
plt.show()

color = ['red', 'pink', 'orange', 'gray']
fig, ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1], marker = 'o', s = 8, c = color[i])
plt.show()

from sklearn.cluster import KMeans
n_clusters = 3
cluster = KMeans(n_clusters = n_clusters, random_state = 0).fit(X)
y_pred = cluster.labels_
print(y_pred)

pre = cluster.fit_predict(X)
pre == y_pred

centroid = cluster.cluster_centers_
print(centroid)
print(centroid.shape)

intertia = cluster.inertia_
print(intertia)

color = ['red', 'pink', 'orange', 'gray']
fig, ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y_pred == i, 0],X[y_pred == i, 1], maker = 'o', s = 8, c = color[i])

ax1.scatter(centroid[:,0], centroid[:,1], marker = 'x', s = 15, c = 'black')

plt.show()

n_clusters = 4

cluster_ = KMeans(n_clusters = n_clusters, random_state = 0).fit(X)
intertia_ = cluster_.inertia_
print(intertia_)






