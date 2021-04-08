import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def preparation(datasetpath):
# In[]
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[1, 2, 3],
                    names=['Rating', 'Helpful', 'Text Review'])

# In[]
    #d = pd.get_dummies(d, columns=['Helpful'])
    #d['Membantu'] = d.apply(lambda row: 1 if (row['Helpful']) >= 1 else 0, axis=1)
    vectorizer = CountVectorizer()    
    d['Recommended'] = d.apply(lambda row: 1 if (row['Rating']) >= 4 else 0, axis=1)
    dvec = vectorizer.fit_transform(d['Text Review'])
    daptarkata = vectorizer.get_feature_names()
    d.head()
    dvec
    
# In[]

    #melakukan pengaacakan data
    d_shuffle = d.sample(frac=1)

# In[]
    #pembagian data training dan data test (70% dan 30%)
    percent_data_training = int(len(d)*0.70)
    
    # In[]
    # pemisahan data training (df_att) dan data test (df_label)
    df_train = d_shuffle.iloc[:percent_data_training]
    df_test = d_shuffle.iloc[percent_data_training:]
    
# In[]
    # data train
    df_train_att = vectorizer.fit_transform(df_train['Text Review'])
# In[]
    # data test
    df_test_att = vectorizer.transform(df_test['Text Review'])
# In[]
    df_train_label = df_train['Recommended']
    df_test_label = df_test['Recommended']
    return df_train_att, df_train_label, df_test_att, df_test_label
# In[]

def training(df_train_att, df_train_label):  
# In[]
    #deklarasi variabel clf untuk klasifikasi menggunakan metode random forest dan max fitur yang digunakan adalah 14
    clf = RandomForestClassifier(n_estimators=100)
    # klasifikasi data training 
    clf = clf.fit(df_train_att, df_train_label)
    return clf
# In[]
def testing(clf, df_test_att):
    return clf.predict(df_test_att)