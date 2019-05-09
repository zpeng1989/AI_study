
#-*- coding:utf-8 -*-
## 自己生成数据并测试

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from matplotlib.backends.backend_pdf import PdfPages


X, y = make_classification(n_samples = 100
                           , n_features = 2
                           , n_redundant = 0
                           , n_informative = 2
                           , random_state = 30
                           , n_clusters_per_class = 1
                        )

print(X.shape)
print(pd.DataFrame(X).head())

rng = np.random.RandomState(2)
X += 2* rng.uniform(size = X.shape)
linearly_separable=[X, y]
#print(linearly_separable.shape)
#print(pd.DataFrame(linearly_separable).head())
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]
print(pd.DataFrame(datasets).shape)
print(pd.DataFrame(datasets).head())
#print(datasets.head())

pdf = PdfPages('cut_figure.pdf')

figure = plt.figure(figsize=(6,9))

i = 1
for ds_index,ds in enumerate(datasets):
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .4, random_state=30)
    x1_min, x1_max = X[:,0].min() - .5, X[:,0].max() + .5
    x2_min, x2_max = X[:,1].min() - .5, X[:,1].max() + .5
    array1, array2 = np.meshgrid(np.arange(x1_min, x1_max,0.2),np.arange(x2_min, x2_max, 0.2))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000','#0000FF'])
    ax = plt.subplot(len(datasets),2,i)
    if ds_index ==0:
        ax.set_title('Input data')
    ax.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = cm_bright,edgecolors = 'k')
    ax.scatter(X_test[:,0], X_test[:,1], c = y_test, cmap = cm_bright, alpha = 0.6, edgecolors = 'k')
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    ax = plt.subplot(len(datasets),2,i)
    clf = DecisionTreeClassifier(max_depth = 5)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    Z = clf.predict_proba(np.c_[array1.ravel(), array2.ravel()])[:, 1]
    Z = Z.reshape(array1.shape)
    ax.contourf(array1, array2, Z, cmap=cm, alpha=.8)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)
    ax.set_xlim(array1.min(), array1.max())
    ax.set_ylim(array2.min(), array2.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_index == 0:
        ax.set_title('Decision tree')
    ax.text(array1.max() - .3, array2.min() + .3,('{:.1f}%'.format(score*100)),size=15, horizontalalignment = 'right')
    i +=1




pdf.savefig()  # 将图片保存在pdf文件中
plt.close()
pdf.close()
