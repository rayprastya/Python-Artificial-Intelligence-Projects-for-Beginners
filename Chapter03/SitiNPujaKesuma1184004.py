import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

def preparation():
    baca = pd.read_csv('Chapter01/dataset/Get Out.csv')
    vc = CountVectorizer()
    ord = OrdinalEncoder()
    baca['Replies'] = ord.fit_transform(baca[['Replies']])
    baca = baca.sample(frac=1)
    baca = [baca[:int(len(baca)*0.75)], baca[int(len(baca)*0.75):]]
    baca = [[vc.fit_transform(baca[0]['comment']),baca[0]['Replies']],[vc.transform(baca[1]['comment']),baca[1]['Replies']]]
    return baca

def training(trainAttr, trainVar):
    t = RandomForestClassifier(n_estimators=80)
    t = t.fit(trainAttr, trainVar)
    return t

def testing(t, testAttr):
    return t.predict(testAttr)





 