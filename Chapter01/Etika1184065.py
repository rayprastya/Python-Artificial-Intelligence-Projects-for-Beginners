import pandas as pd

def preparation(datasetpath):
    d = pd.read_csv(datasetpath, sep=',')
    len(d)
    
    d['popular'] = d.apply(lambda row: 1 if (row['shares']) >= 1000 else 0, axis=1)
    d.head()
    # shuffle rows
    d = d.sample(frac=1)
    # split training and testing data
    d_train = d[:5000]
    d_test = d[5000:]
    
    d_train_att = d_train.drop(['popular'], axis=1)
    d_train_pass = d_train['popular']
    
    d_test_att = d_test.drop(['popular'], axis=1)
    d_test_pass = d_test['popular']
    
    d_att = d.drop(['popular'], axis=1)
    d_pass = d['popular']
    
    return d_train_att,d_train_pass,d_test_att,d_test_pass,d_att,d_pass

def training(d_train_att,d_train_pass):
    # fit a decision tree
    from sklearn import tree
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_pass)
    return t

def testing(t,testdataframe):
    return t.predict(testdataframe)
