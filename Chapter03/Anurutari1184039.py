import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    d = pd.read_csv('Chapter01/dataset/game_review.csv')
    vc = CountVectorizer()
    arr = []
    d['rat'] = d.apply(lambda row: 1 if row['score'] >= 7 else 0, axis=1)
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['review']),d[0]['rat']],[vc.transform(d[1]['review']),d[1]['rat']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)

