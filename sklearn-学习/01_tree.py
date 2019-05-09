
#-*- coding:utf-8 -*-
## scikit-learn study one

##-------------function-----------##
##---- tree.DecisionTreeClassifier
##---- tree.DecisionTreeRegressor
##---- tree.export_graphviz
##---- tree.ExtraTreeClassifier
##---- tree.ExtraTreeRegressor

##--- clf = tree.DecisionTreeClassifier()
##--- clf = clf.fit(X_train, y_train)
##--- result = clf.score(X_test, y_test)

##--- param ----
##--- criterion --纯度方法选择
##-----------------select 1.entropy 2.gini
##--- random_state --随机数选择
##-----------------select int
##--- splitter --剪枝是否随机
##-----------------select 1.random 2.best
##--- maxa_depth --树的深度
##-----------------select int
##--- min_samples_leaf --每一个叶子上最少的样本数
##-----------------select int
##--- min_samples_split --每一个节点允许分支 
##-----------------select int
##--- max_features --最多特征选择
##-----------------select int
##--- min_impurity_decrease --最大信息熵的阈值
##-----------------select flost






from IPython.display import Image
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
#print(wine)
#print(wine.data.shape)
#print(wine.target)

import pandas as pd
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)

#print(wine.feature_names)
#print(wine.target_names)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size = 0.3, random_state=30)

print(Xtrain.shape)
print(Xtest.shape)

clf = tree.DecisionTreeClassifier(criterion="entropy"
                                , random_state = 30
                                , splitter = "random"
                                , max_depth = 3
                                , max_features = 5
                                , min_samples_leaf = 10
                                , min_samples_split = 10
                                , min_impurity_decrease = 0.01
                                , class_weight = 'balanced'#[{0:1,1:5},{0:2,1:3},{0:1,1:6}]
                               )
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest,Ytest)
print(score)
print(clf.score(Xtrain, Ytrain))
print(clf.score(Xtest, Ytest))
print(clf.apply(Xtest))
print(clf.predict(Xtest))



feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜 色强度', '色调', '稀释', '氨酸']

import graphviz
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image

#dot_data = tree.export_graphviz(
#    clf, feature_names=feature_name, out_file ='1.jpg', class_names=["琴酒", "雪莉", "贝尔摩德"], filled=True, rounded = True)
#graph = graphviz.Source(dot_data)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=wine.feature_names,
                     class_names=wine.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph = pydotplus.graph_from_dot_data()
#graph.write_pdf('iris.pdf')


clf.feature_importances_
print([*zip(feature_name, clf.feature_importances_)])


