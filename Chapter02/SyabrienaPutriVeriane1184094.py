import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath,
                    sep=';', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17],
                    names=['gender', 'age', 'class', 'flight distance', 'Inflight wifi service', 'Time Convenient', 'Ease of Online booking', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'satisfaction'])

    d = pd.get_dummies(d, columns=['class'])

    d['jenis kelamin'] = d.apply(lambda row: 1 if (row['gender']) == 'Male' else 0, axis=1)
    d['kepuasan'] = d.apply(lambda row: 1 if (row['satisfaction']) == 'satisfied' else 0, axis=1)
    d = d.drop(['gender', 'satisfaction'], axis = 1)
    d.head()

    #melakukan pengaacakan data
    d_shuffle = d.sample(frac=1)
    
    # pemisahan data training (df_att) dan data test (df_label)
    df_att = d_shuffle.iloc[:, :14]
    df_label = d_shuffle.iloc[:, 13:]
     
    #pembagian data training dan data test (80% dan 20%)
    percent_data_training = int(len(d)*0.70)
    percent_data_test = int(len(d)*0.70)

    # data train
    df_train_att = df_att[:percent_data_training]
    df_train_label = df_label[:percent_data_training]
    
    # data test
    df_test_att = df_att[percent_data_test:]
    df_test_label = df_label[percent_data_test:]

    df_train_label = df_train_label['kepuasan']
    df_test_label = df_test_label['kepuasan']
    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label

def training(df_train_att, df_train_label):  
    from sklearn.ensemble import RandomForestClassifier
    #deklarasi variabel clf untuk klasifikasi menggunakan metode random forest dan max fitur yang digunakan adalah 14
    clf = RandomForestClassifier(max_features=14, random_state=0, n_estimators=100)
    # klasifikasi data training 
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())