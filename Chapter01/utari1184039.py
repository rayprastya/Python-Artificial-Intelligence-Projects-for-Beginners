import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    dframe = pandas.read_csv('Chapter01/dataset/heartfailur.csv', sep=';')
    dframe = dframe.sample(frac=1)
    dframe_train = dframe[:150]
    dframe_test = dframe[150:]
    dframe_train_atrbt = dframe_train.drop(['DEATH_EVENT'], axis=1)
    dframe_train_htfr = dframe_train['DEATH_EVENT']
    dframe_test_atrbt = dframe_test.drop(['DEATH_EVENT'], axis=1)
    dframe_test_htfr = dframe_test['DEATH_EVENT']
    data = [[dframe_train_atrbt,dframe_train_htfr], [dframe_test_atrbt, dframe_test_htfr]]
    return data

def training(dframe_train_atrbt, dframe_train_htfr):
    x = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    x = x.fit(dframe_train_atrbt,dframe_train_htfr)
    return x

def testing(x, testdataframe):
    return x.predict(testdataframe)
