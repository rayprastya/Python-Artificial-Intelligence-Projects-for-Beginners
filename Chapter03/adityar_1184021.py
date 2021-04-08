import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    c = pd.read_csv('Chapter01/dataset/Restaurant.csv')
    vc = CountVectorizer()
    arr = []
    c['suka'] = c.apply(lambda row: 1 if row['Liked'] >=1 else 0, axis=1)
    c = c.sample(frac=1)
    c = [c[:int(len(c)*0.75)], c[int(len(c)*0.75):]]
    data = [[vc.fit_transform(c[0]['Review']),c[0]['suka']],[vc.transform(c[1]['Review']),c[1]['suka']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)

