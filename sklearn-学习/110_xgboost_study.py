# -*- coding: utf-8 -*-

import xgboost as xgb

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time import time
import datetime
import sklearn
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np



data = load_boston()

X = data.data
y = data.target


Xtrain, Xtest, Ytrain, Ytest = TTS(X,y, test_size = 0.3, random_state = 420)

reg = XGBR(n_estimators = 100).fit(Xtrain, Ytain)
print(reg.predict(Xtest))
print(reg.score(Xtest, Ytest))
print(MSE(Ytest, reg.predict(Xtest)))
print(reg.feature_importances_)

reg = XGBR(n_estimators = 100)
CVS(reg, Xtrain, Ytrain, cv = 5).mean()
CVS(reg, Xtrain, Ytrain, cv = 5, scoring = 'neg_mean_squared_error').mean()

sorted(sklearn.metrics.SCORERS.,keys())

rfr = RFR(n_estimators = 100)
CVS(rfr,Xtrain,Ytrain, cv = 5).mean()
CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()


lr = LinearR()
CVS(lr, Xtrain, Ytrain, cv = 5).mean()
CVS(lr, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()

reg = XGBR(n_estimators = 10, silent = False)
CVS(reg, Xtrain, Ytrain, cv = 5, scoring = 'neg_mean_squared_error').mean()


def plot_learning_curve(estimator, title, X, y, ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围 cv=None, #交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, shuffle = True, cv = cv, n_jobs = n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Traing examples")
    ax.set_ylabel("Score")
    ax.grid()
    ax.plot(train_sizes, np.mean(train_scores, axis = 1),'o-', color = 'r', label = 'Training score')
    ax.plot(train_sizes, np.mean(test_scores, axis=1),
            'o-', color='r', label='Test score')
    ax.legend(loc = 'best')
    return ax

cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
plot_learning_curve(XGBR(n_estimators = 100, random_state = 420), 'XGB', Xtrain, Ytrain, ax = None, cv = cv)

plt.show()


    
