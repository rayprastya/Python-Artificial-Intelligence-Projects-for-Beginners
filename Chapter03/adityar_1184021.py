import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    c = pd.read_csv('Chapter01/dataset/data_cholera.csv')
    vc = CountVectorizer()
    arr = []
    c['death'] = c.apply(lambda row: 1 if row['cases_cholera'] >= 100 else 0, axis=1)
    c = c.sample(frac=1)
    c = [c[:int(len(c)*0.80)], c[int(len(c)*0.80):]]
    data = [[vc.fit_transform(c[0]['Country']),c[0]['death']],[vc.transform(c[1]['Country']),c[1]['death']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)

