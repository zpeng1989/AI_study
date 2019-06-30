import pandas as pd


rowdata={'电影名称':['无问西东','后来的我们','前任3','红海行动','唐人街探案','战狼2'],
            '打斗镜头':[1,5,12,108,112,115],
            '接吻镜头':[101,89,97,5,9,8],
            '电影类型':['爱情片','爱情片','爱情片','动作片','动作片','动作片']}

movie_data = pd.DataFrame(rowdata)

print(movie_data)


new_data = [24, 67]
dist = list((((movie_data.iloc[:6, 1:3] - new_data)**2).sum(1))**0.5)
print(dist)


dist_1 = pd.DataFrame({'dist': dist, 'labels':(movie_data.iloc[:6,3])})

dr = dist_1.sort_values(by = 'dist')[:,4]
print(dr)

re = dr.loc[:, 'labels'].value_counts()
print(re)

result = []
result.append(re.index[0])
print(result)

def classofy0(inx, dataset, k):
    result = []
    dist = list((((dataset.iloc[:,1:3]-inx)**2).sum(1))**0.5)
    dist_1 = pd.DataFrame({'dist':dist, 'labels':(dataset.iloc[:,3])})
    dr = dist_1.sort_values(by = 'dist')[:k]
    re = dr.loc[:, 'labels'].value_counts()
    result.append(re.index[0])
    return result

    
