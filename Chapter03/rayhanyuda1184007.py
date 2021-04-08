import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder


def preparation():
    d = pd.read_csv('Chapter01/dataset/hotel_reviews.csv')
    vc = CountVectorizer()
    ord = OrdinalEncoder()
    d['Is_Response'] = ord.fit_transform(d[['Is_Response']])
    d = d.sample(frac=1)
    d = [d[:int(len(d)*0.75)], d[int(len(d)*0.75):]]
    data = [[vc.fit_transform(d[0]['Description']),d[0]['Is_Response']],[vc.transform(d[1]['Description']),d[1]['Is_Response']]]
    return data

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)