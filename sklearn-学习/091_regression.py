# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE



## linear_model.LinearRegression     最小二乘法的线性回归
## linear_model.Ridge
## linear_model.RidgeCV
## linear_model.RidgeClassifier
## linear_model.RidgeClassifierCV
## linear_model.ridge_regression
## linear_model.Lasso
## linear_model.LassoCV
## linear_model.LassoLars
## linear_model.LassoLarsCV
## linear_model.LassoLarsIC
## linear_model.MultiTaskLasso
## linear_model.MultiTaskLassoCV



housevalue = fch()

X = pd.DataFrame(housevalue.data)
y = housevalue.target

print(X.shape)
print(y.shape)
print(X.head())
print(housevalue.feature_names)
X.columns = housevalue.feature_names

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.3, random_state = 420)
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

print(Xtrain.shape)

reg = LR().fit(Xtrain, Ytrain)
yhat = reg.predict(Xtest)
print(yhat)

print(reg.coef_)

#coef_ 
#intercept_

print(MSE(yhat, Ytest))
print(cross_val_score(reg, X, y, cv = 10, scoring = 'mean_squared_error'))





