# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 13:47:59 2021

@author: ANIF
"""

import pandas as pd 
from sklearn.preprocessing import LabelEncoder


def preparation(dataset):
    d = pd.read_csv(dataset,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[1, 2, 3, 4, 5, 6,7],
                    names=['date', 'Temperature', 'Humidity','Light','C02',
                           'HumidityRasio','Occupancy'])
    d = pd.get_dummies(d, columns=['Temperature', 'Humidity','Light','C02'])

    encode = LabelEncoder()
    d['Occupancy'] = encode.fit_transform(d['Occupancy'])
    
    d_shuffle = d.sample(frac=1)
     
    df_att = d_shuffle.iloc[:, 1:4764]
    df_label = d_shuffle.iloc[:, 0:7]

    percent_training = int(len(d)*0.75)
    
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    df_test_att = df_att[percent_training:]
    df_test_label = df_label[percent_training:]

    df_train_label = df_train_label['Occupancy']
    df_test_label = df_test_label['Occupancy']

    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label
    

def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(max_features=1 , random_state=0, n_estimators=100)
    clf = clf.fit(df_train_att, df_train_label)
    return clf


def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())