import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

def train_classifiers(num_of_epochs, dataset, y, test_size, classifiers, print_enabled = False):
    acuracia_tot = []
    tempo_tot = []
    precisao_tot = []
    recall_tot = []
    for i in range(num_of_epochs):
        if print_enabled:
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
        'accuracy': acuracia_tot,
        'time': tempo_tot,
        'precision': precisao_tot,
        'recall': recall_tot
    }