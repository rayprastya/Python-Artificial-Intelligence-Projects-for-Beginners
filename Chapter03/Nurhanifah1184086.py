# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:44:36 2021

@author: User
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def preparation(dataset):
    d = pd.read_csv(dataset)
    
    # generate label class, 1 = positif review, 0 = negatif review   
    d['rating'] = d.apply(lambda row: 1 if row['Score'] >= 2 else 0, axis=1)
                         
    vc = CountVectorizer()
    # shuffle data
    d_shuffle = d.sample(frac=1)
   
    # banyak data
    percent = int(len(d)*0.75)
    # memetakan data train dan data test
    d_train = d_shuffle.iloc[:percent]
    d_test = d_shuffle.iloc[percent:]
    # data train
    d_train_att = vc.fit_transform(d_train['Text']) 
    d_test_att = vc.transform(d_test['Text']) 
    # data test
    d_train_label = d_train['rating']
    d_test_label = d_test['rating']
    return d_train_att, d_train_label, d_test_att, d_test_label

def training(d_train_att, d_train_label):
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max ada 70 independent tree
    clf = RandomForestClassifier(n_estimators=70)
    # fit data train
    clf = clf.fit(d_train_att, d_train_label)
    return clf

def testing(clf, d_test_att):
    return clf.predict(d_test_att)
