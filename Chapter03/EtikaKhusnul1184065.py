# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 23:00:26 2021

@author: ANIF
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def preparation(datasetpath):
    d = pd.read_csv('Chapter01/dataset/OnlineNewsPopularity.csv')
    d['popular'] = d.apply(lambda row: 1 if (row['shares']) >= 1000 else 0, axis=1)
    vectorizer = CountVectorizer()
    d_shuffle = d.sample(frac=1)
    percent = int(len(d)*0.75)
    d_train = d_shuffle.iloc[:percent]
    d_test = d_shuffle.iloc[percent:]
    d_train_att = vectorizer.fit_transform(d_train['url'])
    d_test_att = vectorizer.transform(d_test['url'])
    d_train_label = d_train['popular']
    d_test_label = d_test['popular']
    return d_train_att, d_train_label, d_test_att, d_test_label

def training(d_train_att, d_train_label):
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(d_train_att, d_train_label)
    return clf

def testing(clf, d_test_att):
    return clf.predict(d_test_att)
 