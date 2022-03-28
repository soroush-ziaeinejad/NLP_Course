import pandas as pd
import sklearn_crfsuite
import numpy as np
import eli5
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

df = np.asarray(pd.read_csv('samples.csv'))
training_data = np.zeros((len(df),3))
training_data = []


## DATASET PREPARATION ##
for i,d in enumerate(df):
    tags = []
    points = []
    training_data.append([])
    inputs = d[1][1:-1].split('),')
    for j in inputs:
        inp = f'{j})'
        if inp[0]==' ':
            inp = inp[1:]
        if inp[-2]==')':
            inp = inp[:-1]
        points.append(inp)
        if inp in d[2]:
            training_data[-1].append((inp[1:-1],'1'))
        else:
            training_data[-1].append((inp[1:-1],'0'))
td = np.asarray(training_data)


## FUNCTIONS ##
def word2features(sent, i):
    word = sent[i][0].split(',')
    postag = sent[i][1]
    features = {
        'word1': int(word[0]),      # x of points
        'word2': int(word[1][1:]),  # y of points
        'postag': int(postag)       # is_convexpoint
    }
    return features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [isCP for point, isCP in sent]


## MAIN ##
X = [sent2features(s) for s in td]
y = [sent2labels(s) for s in td]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9995, random_state=42)

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=20,
    all_possible_transitions=False,
    verbose=1
)
crf.fit(X_train, y_train)
print(eli5.format_as_text(eli5.explain_weights(crf)))


#obtaining metrics such as accuracy, etc. on the train set
labels = list(crf.classes_)

ypred = crf.predict(X_train)
print('F1 score on the train set = {}\n'.format(metrics.flat_f1_score(y_train, ypred, average='weighted', labels=labels)))
print('Accuracy on the train set = {}\n'.format(metrics.flat_accuracy_score(y_train, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print('Train set classification report: \n\n{}'.format(metrics.flat_classification_report(
y_train, ypred, labels=sorted_labels, digits=3
)))

#obtaining metrics such as accuracy, etc. on the test set
ypred = crf.predict(X_test)
print('F1 score on the test set = {}\n'.format(metrics.flat_f1_score(y_test, ypred,
average='weighted', labels=labels)))
print('Accuracy on the test set = {}\n'.format(metrics.flat_accuracy_score(y_test, ypred)))

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print('Test set classification report: \n\n{}'.format(metrics.flat_classification_report(y_test, ypred, labels=sorted_labels, digits=3)))
print('len(X_train)',len(X_train))
print('len(y_train)',len(y_train))
print('len(X_test)',len(X_test))
print('len(y_test)',len(y_test))
