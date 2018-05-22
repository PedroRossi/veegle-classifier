import numpy as np
import pandas as pd

def main():
    classifiers1 = [
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge',penalty='l1', max_iter=1000, tol=None))]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MLPClassifier())]),
        Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC())])
    ]

    cv = CountVectorizer(max_df=0.9, max_features=10000)
    classifiers2 = [
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge',penalty='l1', max_iter=1000, tol=None))]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', MLPClassifier())]),
        Pipeline([('vect', cv), ('tfidf', TfidfTransformer()), ('clf', SVC())])
    ]

    col = ['SGC', 'Naive Bayes', 'DecisionTree', 'LogisticRegression', 'MLP', 'SVC']

    obj = train(15, parsed, y, 0.3, classifiers1)
    acuracia = obj['accuracy']
    acuracia_pd = pd.DataFrame(acuracia, columns = col)
    print(np.mean(acuracia_pd))

    obj = train(15, parsed, y, 0.3, classifiers2)
    acuracia = obj['accuracy']
    acuracia_pd = pd.DataFrame(acuracia, columns = col)
    print(np.mean(acuracia_pd))

if __name__ == '__main__':
    main()