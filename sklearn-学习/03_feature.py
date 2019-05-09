#-*- coding:utf-8 -*-
# preprocessing
# min max method

##------ Method Preprocessing --------
# ---StandardScaler         ---标准差
# ---MinMaxScaler           ---均一化
# ---MaxAbsScaler           ---缩放，取得绝对值对大，然后作为分母
# ---RobustScaler           ---四分卫数
# ---Normalizer
# ---PowerTransformer
# ---QuantileTransformer
# ---KernelCenterer

##------ Method Impute ------------
##---Param
##------------missing_value       ---默认np.nan
##------------strategy            ---填充的方法（mean, median, most_frequent, constant）
##------------fill_value          ---使用哪个字符填充
##------------copy                ---是否保留副本

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

data = [[-1, 2], [-0.5, 6], [0, 10], [1,  18]]

new_data = pd.DataFrame(data)
print(data)
print(new_data)
scaler = MinMaxScaler(feature_range=[5,10])
scaler = scaler.fit(data)
result = scaler.transform(data)
print(result)

# preprocessing 
# StandardScaler
# 标准化


data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = StandardScaler()
scaler = scaler.fit(data)
#print(scaler.mean_)
#print(scaler.var_)
x_std = scaler.transform(data)
print(x_std)
print(x_std.mean())
print(x_std.std())

# 缺失值

import pandas as pd
data = pd.read_csv(r'Narrativedata.csv',index_col = 0)

print(data.head())
print(data.info())
print(data.loc[:,'Age'].describe())

Age = data.loc[:,'Age'].values.reshape(-1,1) ##-1表示填充空缺位置，就是默认另一个数字决定数据结构
print(Age[:20])

imp_mean = SimpleImputer()
imp_median = SimpleImputer(strategy='median')
imp_0 = SimpleImputer(strategy='constant',fill_value=0)

imp_mean = imp_mean.fit_transform(Age)
imp_median = imp_median.fit_transform(Age)
imp_0 = imp_0.fit_transform(Age)

print(imp_mean[:20])
print(imp_median[:20])
print(imp_0[:20])



Embarked = data.loc[:,'Embarked'].values.reshape(-1,1)
print(Embarked[:20])
imp_mode = SimpleImputer(strategy = 'most_frequent')
print(imp_mode.fit_transform(Embarked)[:20])


y = data.iloc[:,-1]
le = LabelEncoder()
le = le.fit(y)
label = le.transform(y)
print(y[:20])
print(le.classes_[:20])
print(label[:20])

## preprocessing.OrdinalEncoder
## 特征专用，能够将分类特征转换为分类数值

data.loc[:, "Age"] = imp_median
data.loc[:, "Embarked"] = imp_mode.fit_transform(Embarked)

print('_________________')
data_ = data.copy()
le = OrdinalEncoder()
le = le.fit(data_)
print(le.transform(data_))
print(le.categories_)

#print(data_.head())
#print(OrdinalEncoder().fit().transform(y).categories_)
#print(OrdinalEncoder().fit(data_.iloc[:,1:-1]).categories_)

print('___________________')
print(data.head())
from sklearn.preprocessing import OneHotEncoder
X = data.iloc[:,1:-1]
print(X)
enc = OneHotEncoder(categories= 'auto').fit(X)
result = enc.transform(X).toarray()
print(result)


print(pd.DataFrame(enc.inverse_transform(result)))
print(enc.get_feature_names())
print(result)
print(result.shape)


new_data = pd.concat([data,pd.DataFrame(result)],axis=1)
print(new_data.head())

new_data.drop(['Sex','Embarked'], axis = 1, inplace = True)

new_data.columns = ['Age', 'Survived', 'Female', 'Male', 'Embarked_C', 'Embarkes_Q','Embarked_S']

## 分段function


data2 = data.copy()

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

##------- param
##------- threshold    --cut
##------- n_bins       --num
##------- encode       --encode_method
##------- strategy     --method 

X = data2.iloc[:,0].values.reshape(-1,1)
transformer = Binarizer(threshold = 30).fit_transform(X)
print(transformer[:30])

X = data.iloc[:, 0].values.reshape(-1, 1)
est = KBinsDiscretizer(n_bins = 5, encode = 'ordinal', strategy='uniform')
print(est.fit_transform(X)[:30])
print(set(est.fit_transform(X).ravel()))


est = KBinsDiscretizer(n_bins= 5, encode = 'onehot', strategy='uniform')
print(est.fit_transform(X).toarray()[:30])
