import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from classifier import Classifier
import threading

def test_classifier(name):
    a = Classifier(name, False)
    print('cv_disabled: ' + name + ' - ' + str(np.mean(a.performance['accuracy'])))
    a = Classifier(name, True)
    print('cv_enabled: ' + name + ' - ' + str(np.mean(a.performance['accuracy'])))


def main():
    classifiers = ['sgc', 'naive_bayes', 'decision_tree', 'logistic_regression', 'mlp', 'svc']

    threads = []

    for c in classifiers:
        t = threading.Thread(
            target=test_classifier,
            args=(c,)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

if __name__ == '__main__':
    main()