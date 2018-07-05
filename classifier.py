from .parser import get_text_from_html
from .train import train_classifiers
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class Classifier:

    def __init__(self, name, cv_enabled, parsed, evaluation):
        self.name = name
        self.cv = CountVectorizer(max_df=0.9, max_features=10000) if cv_enabled else CountVectorizer()
        self.parsed = parsed
        self.evaluation = evaluation
        self.classifier = None
        self.performance = None
        self.train()

    def train(self):
        pipeline = []
        pipeline.append(('vect', self.cv))
        pipeline.append(('tfidf', TfidfTransformer()))
        classifiers_dicitionary = {
            'sgc': MultinomialNB(),
            'naive_bayes': SGDClassifier(loss='hinge',penalty='l1', max_iter=1000, tol=None),
            'decision_tree': DecisionTreeClassifier(),
            'logistic_regression': LogisticRegression(),
            'mlp': MLPClassifier(),
            'svc': SVC()
        }
        if self.name not in classifiers_dicitionary:
            raise Exception('Classifier not found')
        pipeline.append(('clf', classifiers_dicitionary[self.name]))
        self.classifier = Pipeline(pipeline)
        data = train_classifiers(15, self.parsed, self.evaluation, 0.3, [self.classifier])
        self.performance = data

    def predict_from_file_path(self, path):
        text = get_text_from_html(path)
        return self.classifier.predict([text])[0]