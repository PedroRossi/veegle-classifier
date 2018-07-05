from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from parser import parse_samples
from classifier import Classifier
import numpy as np
import threading
import os

parsed = []
y = []

def load_data():
    parse_samples('pages/positivos', parsed, y, 1)
    parse_samples('pages/negativos', parsed, y, 0)

def test_classifier(name, parsed, y):
    a = Classifier(name, False, parsed, y)
    print('cv_disabled: ' + name + ' - ' + str(np.mean(a.performance['accuracy'])))
    a = Classifier(name, True, parsed, y)
    print('cv_enabled: ' + name + ' - ' + str(np.mean(a.performance['accuracy'])))

def test():
    classifiers = ['sgc', 'naive_bayes', 'decision_tree', 'logistic_regression', 'mlp', 'svc']

    threads = []

    for c in classifiers:
        t = threading.Thread(
            target=test_classifier,
            args=(c, parsed, y)
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

def main():
    test()
    return
    classifier = Classifier('naive_bayes', False, parsed, y)
    print('Accuracy: '+str(np.mean(classifier.performance['accuracy'])))
    print('Time: '+str(np.mean(classifier.performance['time'])))
    print('Precision: '+str(np.mean(classifier.performance['precision'])))
    print('Recall: '+str(np.mean(classifier.performance['recall'])))
    path = '../domains/domains-hr/'
    for domain in os.listdir(path):
        full_path = os.path.join(path, domain, '1')
        for filename in os.listdir(full_path):
            if not filename.endswith('.html'):
                continue
            file_path = os.path.join(full_path, filename)
            result = classifier.predict_from_file_path(file_path)
            new_good_path = os.path.join(full_path, '1')
            new_bad_path = os.path.join(full_path, '0')
            dest_path = None
            if result:
                if not os.path.exists(new_good_path):
                    os.mkdir(new_good_path)
                dest_path = os.path.join(new_good_path, filename)
            else:
                if not os.path.exists(new_bad_path):
                    os.mkdir(new_bad_path)
                dest_path = os.path.join(new_bad_path, filename)
            source = open(file_path, 'r')
            dest = open(dest_path, 'w')
            dest.write(source.read())
            source.close()
            dest.close()
            if result:
                print(domain + ' - ' + filename)

if __name__ == '__main__':
    load_data()
    main()
