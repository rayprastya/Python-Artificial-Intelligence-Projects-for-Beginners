import pandas
from sklearn import tree
from sklearn.model_selection import cross_val_score

def zoo():
    animal = pandas.read_csv('Chapter01/dataset/zoo.csv', sep=',')
    animal = animal.sample(frac=1)
    animal_train = animal[:50]
    animal_test = animal[50:]
    animal_train_attribute = animal_train.drop(['type'], axis=1)
    animal_train_type = animal_train['type']
    animal_test_attribute = animal_test.drop(['type'], axis=1)
    animal_test_type = animal_test['type']
    data = [[animal_train_attribute,animal_train_type], [animal_test_attribute, animal_test_type]]
    return data

def latihan(animal_train_att, animal_train_type):
    hasil = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    hasil = hasil.fit(animal_train_att,animal_train_type)
    return hasil

def percobaan(hasil, testdataframe):
    return hasil.predict(testdataframe)