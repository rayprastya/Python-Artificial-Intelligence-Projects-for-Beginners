import pandas as pd

#preprocessing dataset
def preparation(datasetpath):
    #buka dataset pake pandas
    d = pd.read_csv(datasetpath,
                    sep=',', header=None, error_bad_lines=False,
                    warn_bad_lines=False, usecols=[0, 1, 2, 3, 4, 5],
                    names=['Class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor'])
    #buat data dummy
    d = pd.get_dummies(d, columns=['cap-shape', 'cap-surface', 'cap-color', 'odor'])
    
    #mengubah ke nilai 1 dan 0 (true dan false)
    d['Bruises'] = d.apply(lambda row: 1 if (row['bruises']) == 't' else 0, axis=1)
    d['Bisa Dimakan'] = d.apply(lambda row: 1 if (row['Class']) == 'e' else 0, axis=1)
    d = d.drop(['Class', 'bruises'], axis = 1)
    d.head()
    
    #acak data
    d_shuffle = d.sample(frac=1)

    #pemisahan training (df_att) dan test(df_label)
    df_att = d_shuffle.iloc[:, :29]
    df_label = d_shuffle.iloc[:, 29:]
    
    percent_training = int(len(d)*0.80)
    percent_test = int(len(d)*0.20)
    
    # data train
    df_train_att = df_att[:percent_training]
    df_train_label = df_label[:percent_training]
    
    # data test
    df_test_att = df_att[percent_test:]
    df_test_label = df_label[percent_test:]

    df_train_label = df_train_label['Bisa Dimakan']
    df_test_label = df_test_label['Bisa Dimakan']
    
    return df_train_att, df_train_label, df_test_att, df_test_label, df_att, df_label

#training dataset
def training(df_train_att, df_train_label):
    from sklearn.ensemble import RandomForestClassifier
    # penetapan variabel clf sebagai variabel klasifikasi random forest deng max. fitur 10 pada tiap tree nya
    clf = RandomForestClassifier(max_features=10, random_state=0, n_estimators=100)
    # klasifikasi data training 
    clf = clf.fit(df_train_att, df_train_label)
    return clf

def testing(clf, df_test_att):
    return clf.predict(df_test_att.head())