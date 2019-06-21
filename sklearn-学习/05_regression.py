#-*- coding:utf-8 -*-
## scikit-learn linear_model
## linear_model
## linear_model.LogisticRegression              logit回归，最大熵分类器
## linear_model.LogisticRegressionCV            带有交叉熵的分类器
## linear_model.Logistic_regression_path        正则化参数列表
## linear_model.SGDClassifier                   梯度下降求解分类器
## linear_model.SGDRegressor                    梯度下降线性回归模型
## metrics.log_loss                             对数损失，交叉熵损失


## 评估标准
## metric.confusion_matrix                      混淆矩阵
## metric.roc_auc_score                         ROC曲线及auc结果
## metric.accuracy_score                        精确性


## 参数(正则化参数)
## penalty l1 和 l2 正则化
## C 正则化系数的倒数


from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

print(data.data.shape)


lrl1 = LR(penalty = "l1", solver = "liblinear", C = 0.5, max_iter = 1000)
lrl2 = LR(penalty = "l2", solver = "liblinear", C = 0.5, max_iter = 1000)

lrl1 = lrl1.fit(X, y)
print(lrl1.coef_)


lrl2 = lrl2.fit(X, y)
print(lrl2.coef_)


l1 = []
l2 = []
l1test = []
l2test = []

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size = 0.3, random_state = 420)

for i in np.linspcae(0.05, 1.19):
    lrl1 = LR(penalty = "l1", solver = "liblinear", C = i, max_iter = 1000)
    lrl2 = LR(penalty = "l2", solver = "liblinear", C = i, max_iter = 1000)
    lrl1 = lrl1.fit(Xtrain, Ytrain)
    l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
    l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))
    lrl2 = lrl2.fit(Xtrain, Ytrain)
    l2.append(accuracy_sorce(lrl2.predict(Xtrain), Ytrain))
    l2test.append(accuarcy_score(lrl2.predict(Xtest), Ytest))

#graph = [l1, l2, l1test, l2test]




