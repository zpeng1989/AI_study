# -*- coding: utf-8 -*-

## naive_bayes.BernoulliNB
## naive_bayes.GaussianNB
## naive_bayes.MultinomialNB
## naive_bayes.ComplementNB
## linear_model.BayesianRidge

## param
## prior 先验概率
## var_smoothing 浮点数

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,y, test_size = 0.3, random_state = 420)

print(Xtrain[0:5,:])
print(Ytrain[0:5])
gnb = GaussianNB().fit(Xtrain, Ytrain)
acc_score = gnb.score(Xtest, Ytest)
Y_pred = gnb.predict(Xtest)
print(Y_pred.shape)
prob = gnb.predict_proba(Xtest)
print(prob.shape)
print(prob)

from sklearn.metrics import confusion_matrix as CM
print(CM(Ytest, Y_pred))


