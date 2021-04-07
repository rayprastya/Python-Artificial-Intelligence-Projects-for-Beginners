# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 22:05:54 2021

@author: Dinda Majesty
"""
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def preparation(dataset):
    d = pd.read_csv(dataset)

    d.drop(['Top Ten'], axis=1)

    le = preprocessing.LabelEncoder()

    d['Stars'] = le.fit_transform(d['Stars'].astype(float))

    #3: Very Recommended, 2: Recommended, 1: less recommended, 0: not recommended
    d['Recommended'] = d.apply(lambda row: 3 if row['Stars'] >= 8 <= 10 else 2 if row['Stars'] >= 5 <= 7 else 1 if row['Stars'] >= 2 <= 4 else 0 , axis=1)

    vc = CountVectorizer()

    d_shuffle = d.sample(frac=1)

    data = int(len(d) * 0.75)

    d_train = d_shuffle.iloc[:data]
    d_test = d_shuffle.iloc[data:]

    d_train_att = vc.fit_transform(d_train['Variety'])
    d_test_att = vc.transform(d_test['Variety'])

    d_train_label = d_train['Recommended']
    d_test_label = d_test['Recommended']
    return d_train_att, d_train_label, d_test_att, d_test_label


def training(d_train_att, d_train_label):
    clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=50)
    clf = clf.fit(d_train_att, d_train_label)
    return clf


def testing(clf, d_test_att):
    return clf.predict(d_test_att)
