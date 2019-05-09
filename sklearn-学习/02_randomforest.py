#-*- coding:utf-8 -*-

##-----------------------function------------------##
##--- ensemble.AdaBoostClassifier
##--- ensemble.AdaBoostRegressor
##--- ensemble.BaggingClassifier
##--- ensemble.BaggingRegressor
##--- ensemble.ExtraTreesClassifier
##--- ensemble.ExtraTreesRegressor
##--- ensemble.GradientBoostingClassifier
##--- ensemble.GradientBoostingRegressor
##--- ensemble.IsolationForest
##--- ensemble.RandomForestClassifier
##--- ensemble.RandomForestRegressor
##--- ensemble.RandomTreeEmbedding
##--- ensemble.VotingClassifier

##-----------------------param----------------------##
##--- criterion --纯度方法选择
##-----------------select 1.entropy 2.gini
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
##--- n_estimators --树木数量选择
##-----------------select int

#rfc = RandomForestClassifier()
#rfc = rfc.fit(X_train, y_train)
#reslut = rfc.score(X_test, y_test)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
wine = load_wine()
print(wine.data)
print(wine.target)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size = 0.3, random_state = 30)

clf = DecisionTreeClassifier(random_state = 30)
rfc = RandomForestClassifier(random_state = 30)

clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)

score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)

print('Single tree;{}'.format(score_c), 'Random Forest:{}'.format(score_r))

rfc_l = []
clf_l = []

for i in range(10):
    rfc = RandomForestClassifier(n_estimators = 25)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv = 10).mean()
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target, cv = 10).mean()
    clf_l.append(clf_s)

pdf = PdfPages('RandomForest1.pdf')

plt.plot(range(1,11),rfc_l, label = 'Random Forest')
plt.plot(range(1,11),clf_l, label = 'Decision Tree')
plt.legend()

pdf.savefig()  # 将图片保存在pdf文件中
plt.close()
pdf.close()

superpa = []

for i in range(200):
    rfc = RandomForestClassifier(n_estimators = i+1)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv = 10).mean()
    superpa.append(rfc_s)
print(max(superpa), superpa.index(max(superpa)))

pdf = PdfPages('RandomForest2.pdf')
plt.figure(figsize = [20, 5])
plt.plot(range(1,201),superpa)
pdf.savefig()
plt.close()
pdf.close()

rfc = RandomForestClassifier(n_estimators=40)
rfc = rfc.fit(Xtrain, Ytrain)
print(rfc.score(Xtest, Ytest))
print(rfc.feature_importances_)
print(rfc.apply(Xtest))
print(rfc.predict(Xtest))
print(rfc.predict_proba(Xtest))
print(rfc.estimators_)
#print(rfc.oob_score_)

