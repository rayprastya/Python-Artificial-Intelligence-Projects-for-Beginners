import pandas as pd
from sklearn import tree


#preprocessing dataset
def preprocess(dataset):
    #ambil data dari file csv (buka file csv)
    d = pd.read_csv(dataset)
    len(d)
    d.head()
    
    #acak baris 
    d = d.sample(frac=1)
    
    #Pembagian training data (80%) dan test data (20%)
    persen_train = int(len(d)*0.80)
    persen_test = int(len(d)*0.20)
    d_train = d[:persen_train]
    d_test = d[persen_test:]
    
    d_train_att = d_train.drop(['DEATH_EVENT'], axis=1)
    d_train_death = d_train['DEATH_EVENT']
    
    d_test_att = d_test.drop(['DEATH_EVENT'], axis=1)
    d_test_death = d_test['DEATH_EVENT']
    
    d_att = d.drop(['DEATH_EVENT'], axis=1)
    d_death = d['DEATH_EVENT']
    
    d_death = d['DEATH_EVENT']
    
    return d_train_att, d_train_death, d_test_att, d_test_death, d_att, d_death


def train(d_train_att, d_train_death):
    t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    t = t.fit(d_train_att, d_train_death)
    return t

def test(t, testdataframe):
    return t.predict(testdataframe)


