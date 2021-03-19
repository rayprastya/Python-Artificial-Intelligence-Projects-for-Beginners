import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def preparation():
    cancer = pandas.read_csv('Chapter01/dataset/haberman.data.csv', sep=',')
    cancer = cancer.sample(frac=1)
    cancer_train = cancer[:151]
    cancer_test = cancer[153:]
    cancer_train_attribute = cancer_train.drop(['live'], axis=1)
    cancer_train_live = cancer_train['live']
    cancer_test_attribute = cancer_test.drop(['live'], axis=1)
    cancer_test_live = cancer_test['live']
    data = [[cancer_train_attribute,cancer_train_live], [cancer_test_attribute, cancer_test_live]]
    return data

def training(cancer_train_att, cancer_train_win):
    tr = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tr = tr.fit(cancer_train_att,cancer_train_win)
    return tr

def testing(tr, testdataframe):
    return tr.predict(testdataframe)