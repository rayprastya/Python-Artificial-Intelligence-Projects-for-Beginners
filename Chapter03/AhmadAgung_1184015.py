# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:07:34 2021

@author: ahmad
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def preparation(datasetmovie):
    mv = pd.read_csv(datasetmovie )
    
    mv['class'] = mv.apply(lambda row: 1 if row['imdb'] >= 7 else 0, axis=1)
    vc = CountVectorizer()
    
    mv_shuffle = mv.sample(frac=1)
    percent = int(len(mv)*0.95)
    
    mv_train = mv_shuffle.iloc[:percent]
    mv_test = mv_shuffle.iloc[percent:]
    
    mv_train_att = vc.fit_transform(mv_train['movie']) 
    mv_test_att = vc.transform(mv_test['movie']) 
    
    mv_train_label = mv_train['class']
    mv_test_label = mv_test['class']
    
    return mv_train_att, mv_train_label, mv_test_att, mv_test_label


def training(mv_train_att, mv_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(mv_train_att, mv_train_label)
    return clf

def testing(clf, mv_test_att):
    return clf.predict(mv_test_att)