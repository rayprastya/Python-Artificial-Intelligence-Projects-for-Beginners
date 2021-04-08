import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    v = pd.read_csv('Chapter01/dataset/Clothing_Reviews.csv')
    vc = CountVectorizer()
    arr = []
    v['poin'] = v.apply(lambda row: 1 if row['Rating'] >=3 else 0, axis=1)
    v = v.sample(frac=1)
    v = [v[:int(len(v)*0.75)], v[int(len(v)*0.75):]]
    data = [[vc.fit_transform(v[0]['Review Text']),v[0]['poin']],[vc.transform(v[1]['Review Text']),v[1]['poin']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)
