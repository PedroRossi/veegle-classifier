import os
import time
import numpy as np
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif


def parse_samples(path, parsed_list, y, type_of_sample):
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        if fullpath.endswith('.html'):
            page = BeautifulSoup(open(fullpath),'html.parser')
            table = page.find_all('p')
            s = ""
            p = []
            for x in table:
                if(x.text != 'placeholder'):
                    s += x.text + " "
            s=''.join(i for i in s if  not i.isdigit())
            parsed_list.append(s)
            y.append(type_of_sample)

def train(num_of_epochs, dataset, y, test_size, classifiers):
    acuracia_tot = []
    tempo_tot = []
    precisao_tot = []
    recall_tot = []
    for i in range(num_of_epochs):
        print ('Epoch: ', (i+1))
        x_train, x_test, y_train, y_test = train_test_split(dataset, y, test_size=test_size, random_state=None)
        acuracia = []
        tempo = []
        precisao = []
        recall = []
        for classifier in classifiers:
            time1 = time.time()
            classifier.fit(x_train, y_train)
            time2 = time.time()
            y_pred = classifier.predict(x_test)
            score = classifier.score(x_test, y_test)
            precision = precision_score(y_test, y_pred, average='weighted', labels=np.unique(y_pred))
            rec = recall_score(y_test, y_pred,  average='weighted', labels=np.unique(y_pred))
            acuracia.append(score)
            tempo.append(time2-time1)
            precisao.append(precision)
            recall.append(rec)
        acuracia_tot.append(acuracia)
        tempo_tot.append(tempo)
        precisao_tot.append(precisao)
        recall_tot.append(recall)
    return {
        'acuracia': acuracia_tot,
        'tempo': tempo_tot,
        'precisao': precisao_tot,
        'recall': recall_tot
    }

def main():
    parsed = []
    y = []

    parse_samples('pages/positivos', parsed, y, 1)
    parse_samples('pages/negativos', parsed, y, 0)


    skf = StratifiedKFold(n_splits=15)
    classifiers1 = [
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge',penalty='l1', max_iter=1000, tol=None))]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MLPClassifier())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC())])
    ]

    cv = CountVectorizer (max_df=0.9, max_features=10000)
    classifiers2 = [
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge',penalty='l1', max_iter=1000, tol=None))]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', MLPClassifier())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', SVC())])
    ]
    train(15, parsed, y, 0.3, classifiers1)
    train(15, parsed, y, 0.3, classifiers2)

if __name__ == '__main__':
    main()