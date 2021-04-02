import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def preparation():
    dt = pd.read_csv('Chapter01/dataset/heartfailur.txt', sep=',', usecols=[0,1,2,3,4,5,6,7,8], header=None, names=['age', 'sex', 'Cough', 'dyspnea', 'edema', 'Nausea', 'increasedheartrate', 'thinkingdisorder', 'heartfailur'])
   
    dt = dt.sample(frac=1)
    dt = [dt.iloc[:,:8], dt.iloc[:, 8:]]


    dt_atribut = dt.pop(0)
    dt_varr = dt.pop(0)
    

    length = int(len(dt_varr)*0.75)

    trn_Varr = dt_varr[:length]
    trn_atribut = dt_atribut[:length]

    test_Varr = dt_varr[length:]
    test_atribut = dt_atribut[length:]

    return [[trn_atribut, trn_Varr], [test_atribut, test_Varr]]

def training(trn_atribut, trn_Varr):
    t = RandomForestClassifier(max_features=4, random_state=0, n_estimators=100)
    t = t.fit(trn_atribut, trn_Varr)
    return t

def testing(t, test_atribut):
    return t.predict(test_atribut)