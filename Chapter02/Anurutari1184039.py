import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation():
    data = pd.read_csv('Chapter01/dataset/heartfailur.txt', sep=',', usecols=[0,1,2,3,4,5,6,7,8], header=None, names=['age', 'sex', 'Cough', 'dyspnea', 'edema', 'Nausea', 'increasedheartrate', 'thinkingdisorder', 'heartfailur'])
   
    data = data.sample(frac=1)
    data = [data.iloc[:,:8], data.iloc[:, 8:]]


    data_attribut = data.pop(0)
    data_varr = data.pop(0)
    

    length = int(len(data_varr)*0.75)

    train_Varr = data_varr[:length]
    train_attribut = data_attribut[:length]

    test_Varr = data_varr[length:]
    test_attribut = data_attribut[length:]

    return [[train_attribut, train_Varr], [test_attribut, test_Varr]]

def training(train_attribut, train_Varr):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(train_attribut, train_Varr)
    return t

def testing(t, test_attribut):
    return t.predict(test_attribut)