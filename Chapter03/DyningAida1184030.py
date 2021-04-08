# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:25:54 2021

@author: DyningAida
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def preparation(dataset):
    d = pd.read_csv(dataset)
    
    # generate label class, 2 = sangat baik, 1 = baik, 0 = kurang baik    
    d['class'] = d.apply(lambda row: 2 if row['rating'] >= 9 else
                         ( 1 if row['rating'] >= 7 else 0), axis=1)
    vc = CountVectorizer()
    # shuffle data
    d_shuffle = d.sample(frac=1)
    # banyak data
    percent = int(len(d)*0.75)
    # memetakan data train dan data test
    d_train = d_shuffle.iloc[:percent]
    d_test = d_shuffle.iloc[percent:]
    # data train
    d_train_att = vc.fit_transform(d_train['book_title']) # fit book-title on training set
    d_test_att = vc.transform(d_test['book_title']) # reuse on testing set
    # data test
    d_train_label = d_train['class']
    d_test_label = d_test['class']
    return d_train_att, d_train_label, d_test_att, d_test_label


def training(d_train_att, d_train_label):
    # instansiasi variabel klasifikasi dengan metode random forest classifier, max ada 50 independent tree
    clf = RandomForestClassifier(n_estimators=50)
    # fit data train
    clf = clf.fit(d_train_att, d_train_label)
    return clf

def testing(clf, d_test_att):
    return clf.predict(d_test_att)
