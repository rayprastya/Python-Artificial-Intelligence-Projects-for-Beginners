import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation():
    data = pd.read_csv('Chapter01/dataset/lungcancer.txt', sep=',', usecols=[0,1,2,3,4,5,6,7], header=None, names=['age', 'gender', 'chroniccough', 'coughingupblood', 'drasticweightloss', 'chestandbonepain', 'difficultybreathing', 'lungcancer'])
   
    data = data.sample(frac=1)
    data = [data.iloc[:,:7], data.iloc[:, 7:]]


    data_atribut = data.pop(0)
    data_var = data.pop(0)
    

    length = int(len(data_var)*0.75)

    train_Var = data_var[:length]
    train_atribut = data_atribut[:length]

    test_Var = data_var[length:]
    test_atribut = data_atribut[length:]

    return [[train_atribut, train_Var], [test_atribut, test_Var]]

def training(train_atribut, train_Var):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(train_atribut, train_Var)
    return t

def testing(t, test_atribut):
    return t.predict(test_atribut)