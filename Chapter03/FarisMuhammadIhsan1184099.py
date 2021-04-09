import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation(datasetpath):
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[2, 3],
                    names=['Review_singkat', 'Label'])

    vectorizer = CountVectorizer()
    d.head()
    
    #melakukan pengaacakan data
    d_shuffle = d.sample(frac=1)

    #pembagian data training dan data test (70% dan 30%)
    percent_data_training = int(len(d)*0.70)
    

    # pemisahan data training (df_att) dan data test (df_label)
    df_train = d_shuffle.iloc[:percent_data_training]
    df_test = d_shuffle.iloc[percent_data_training:]
    
    # data train
    df_train_att = vectorizer.fit_transform(df_train['Review_singkat'])

    # data test
    df_test_att = vectorizer.transform(df_test['Review_singkat'])

    df_train_label = df_train['Label']
    df_test_label = df_test['Label']
    return df_train_att, df_train_label, df_test_att, df_test_label


def training(df_train_att, df_train_label):  
    #deklarasi variabel clf untuk klasifikasi menggunakan metode random forest 
    clf = RandomForestClassifier(n_estimators=100)
    # klasifikasi data training 
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att)