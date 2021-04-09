# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:33:34 2021

@author: Lenovo
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',', encoding='latin-1')
    
    d['Satisfied'] = d.apply(lambda row: 1 if (row['Rating']) >= 3 else 0, axis=1)
    
    vectorizer = CountVectorizer()
    
    # shuffle rows
    d = d.sample(frac=1)
    
    d_train=d.iloc[:21000]
    d_test=d.iloc[21000:]
    
    # split training and testing data
    d_train_att=vectorizer.fit_transform(d_train['Review_Text'])
    d_test_att=vectorizer.transform(d_test['Review_Text'])
    
    d_train_label=d_train['Satisfied']
    d_test_label=d_test['Satisfied']

    return d_train_att, d_train_label, d_test_att, d_test_label

def training(d_train_att, d_train_label):
    clf = RandomForestClassifier(max_features=50, random_state=0, n_estimators=50) 
    clf = clf.fit(d_train_att, d_train_label)
    return clf

def testing(clf, d_test_att):
    return clf.predict(d_test_att)


    

