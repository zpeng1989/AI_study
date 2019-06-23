# -*- coding: utf-8 -*-

from sklearn.svm import SVC

#clf = SVC()
#clf = clf.fit(X_train, y_train)
#result = clf.score(X_test, y_test)

# linear          线性核  
# poly            多项式核
# sigmoid         双曲正则函数核
# rbf             高斯径向核

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

data = load_breast_cancer()

X = data.data
y = data.target

print(X.shape)
print(np.unique(y))

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y,test_size = 0.3, random_state = 420)
Kernel = ["linear", "poly", "rbf", "sigmoid"]

'''
for kernel in Kernel:
    time0 = time()
    clf = SVC(kernel = kernel, gamma = 'auto', cache_size = 5000).fit(Xtrain, Ytrain)
    print(kernel, clf.score(Xtest,Ytest))
    print(datetime.datetime.fromtimestamp(time()-time0))

'''

score = []
gamma_range = np.logspace(-10,1,50)

for i in gamma_range:
    clf = SVC(kernel = 'rbf', gamma = i, cache_size= 50000).fit(Xtrain, Ytrain)
    score.append(clf.score(Xtest, Ytest))

print(max(score), gamma_range[score.index(max(score))])

#plt.plot(gamma_range, score)
#plt.show()

gamma_range = np.logspace(-10, 1, 20)
coef0_range = np.linspace(0, 5, 10)

param_grid = dict(gamma = gamma_range, coef0 = coef0_range)

cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 420)
grid = GridSearchCV(SVC(kernel="poly", degree=1, cache_size=5000), param_grid = param_grid, cv = cv)

grid.fit(X,y)

print(grid.best_params_, grid.best_score_)










