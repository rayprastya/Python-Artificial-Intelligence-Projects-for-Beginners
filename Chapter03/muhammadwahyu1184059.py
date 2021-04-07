import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation():
    d = pd.read_csv('Chapter01/dataset/train.csv')
    vc = CountVectorizer()
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['user_review']),d[0]['user_suggestion']],[vc.transform(d[1]['user_review']),d[1]['user_suggestion']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)