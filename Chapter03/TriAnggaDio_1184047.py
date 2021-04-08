import pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


def running_chapter_3(filename='Chapter01/dataset/indonesian_hate_speech.csv'):
    # preprocessing
    df = pandas.read_csv(filename)

    vectorizer = CountVectorizer()

    df_shuffle = df.sample(frac=1)

    separator_data_train_and_test = int(len(df.index) * 0.80)

    d_training = df_shuffle[:separator_data_train_and_test]
    d_testing = df_shuffle[separator_data_train_and_test:]

    d_training_att = vectorizer.fit_transform(d_training['Tweet'])
    d_testing_att = vectorizer.transform(d_testing['Tweet'])

    d_training_label = d_training['HS']
    d_testing_label = d_testing['HS']

    # training
    clf = RandomForestClassifier(n_estimators=80)
    clf.fit(d_training_att, d_training_label)

    # predicting and scoring
    predict_results = clf.predict(d_testing_att)
    model_score_percentage = int(clf.score(d_testing_att, d_testing_label) * 100)
    model_scoring = cross_val_score(clf, d_training_att, d_training_label, cv=5)

    print(f"Hasil Prediksi: {predict_results}")
    print(f"Model Percentage: {model_score_percentage}%")
    print(f"Confusion Matrix: {confusion_matrix(d_testing_label, predict_results)}")
    print(f"Score Rata-Rata: {model_scoring.mean()}")
    print(f"Score Standar Deviasi: {model_scoring.std()}")

    return predict_results[0]