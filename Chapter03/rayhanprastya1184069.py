import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rc
from sklearn.feature_extraction.text import CountVectorizer as cv

def preparation():
    d = pd.read_csv('Chapter01/dataset/FIFA_20.csv')
    vc = cv()
    d['stat'] = d.apply(lambda row: 1 if row['overall'] >= 85 else 0 , axis=1)
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['name']),d[0]['stat']],[vc.transform(d[1]['name']),d[1]['stat']]]
    return data

def training(trainAttr, trainVar):
    t = rc(n_estimators=50)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)
