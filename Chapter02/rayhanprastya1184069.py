import pandas as pd
from sklearn.ensemble import RandomForestClassifier as rf

def preparation(dataset):
    data = pd.read_csv(dataset, sep=';', usecols=[1,2,3,4,5,6,7,8,9,10,11,12], header=None, names=['id','age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active','cardio'])
    data = data.sample(frac=1)
    data = [data.iloc[:,:11], data.iloc[:,11:]]

    atr = data.pop(0)
    var = data.pop(0)

    pj = int(len(var)*0.75)

    trainvar = var[:pj]
    trainatr = atr[:pj]
    testvar = var[pj:]
    testatr = atr[pj:]

    return [[trainatr, trainvar], [testatr, testvar]] 

def training(trainatr, trainvar):
    a = rf(max_features=4, random_state=0, n_estimators=100)
    a = a.fit(trainatr, trainvar)
    return a

def testing(a, testatr):
    return a.predict(testatr)
    

