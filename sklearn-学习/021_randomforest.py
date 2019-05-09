from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

data = load_breast_cancer()
print(data.data)
print(data.data.shape)
print(data.target)
print(data.target.shape)

rfc = RandomForestClassifier(n_estimators = 100, random_state = 30)
score_pre = cross_val_score(rfc, data.data, data.target, cv =10).mean()
print(score_pre)

scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators = i+1
                                , random_state =30
                                )
    score = cross_val_score(rfc, data.data, data.target, cv = 10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)

pdf = PdfPages('RandomForest3.pdf')
plt.figure(figsize = [20,5])
plt.plot(range(1,201,10),scorel)
pdf.savefig()
plt.close()
pdf.close()

param_grid = {'max_depth': np.arange(1, 20, 1), 'criterion': ['gini', 'entropy'], 'min_samples_split': np.arange(2, 2+20,1),'min_samples_leaf':np.arange(1,1+10,1),'max_features':np.arange(5,30,1)}

rfc = RandomForestClassifier(n_estimators = 40
                            , random_state = 30
                            )
GS = GridSearchCV(rfc, param_grid, cv = 10)
GS.fit(data.data, data.target)
print(GS.best_params_)
print(GS.best_score_)





#pdf = PdfPages('RandomForest2.pdf')

#pdf.savefig()
#plt.close()
#pdf.close()


