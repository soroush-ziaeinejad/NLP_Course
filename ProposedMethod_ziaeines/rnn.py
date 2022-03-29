import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, TimeDistributed, Embedding, Bidirectional, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
df = np.asarray(pd.read_csv('samples.csv'))
X = []
Y = []

## DATASET PREPARATION ##
for i,d in enumerate(df):
    tags = []
    points = []
    X.append([])
    Y.append([])
    inputs = d[1][1:-1].split('),')
    for j in inputs:
        inp = f'{j})'
        if inp[0]==' ':
            inp = inp[1:]
        if inp[-2]==')':
            inp = inp[:-1]
        points.append(inp)
        if inp in d[2]:
            X[-1].append(list(eval(inp)))#[1:-1])
##            X[-1].append([1,1])
            Y[-1].append(1)
        else:
            X[-1].append(list(eval(inp)))#[1:-1])
##            X[-1].append([0,0])
            Y[-1].append(0)
##X = np.asarray(X,dtype=object)
##Y = np.asarray(Y,dtype=object)

print('sample X: ', X[0], '\n')
print('sample Y: ', Y[0], '\n')

print('Length of first input sequence : {}'.format(len(X[0])))
print('Length of first output sequence : {}'.format(len(Y[0])))

## unique words ##
UX = []
maxseqlen = 0
for x in X:
    if len(x)>maxseqlen:
        maxseqlen = len(x)
    for xx in x:
        if xx not in UX:
            UX.append(xx)
VOCABULARY_SIZE = len(UX)
MAX_SEQ_LENGTH = maxseqlen

X_padded = pad_sequences(X, maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='post')
Y_padded = pad_sequences(Y, maxlen=MAX_SEQ_LENGTH, padding='pre', truncating='post')
X_padded = np.float32(X_padded)
Y_padded = np.float32(Y_padded)
# print the first sequence
print(X_padded[0], "\n"*3)
print(Y_padded[0])




X_train, X_test, y_train, y_test = train_test_split(X_padded, Y_padded,
                                                    test_size=0.50, random_state=42)




NUM_CLASSES = 2
EMBEDDING_SIZE = 5

'''
## VANILLA RNN ##
# create architecture
rnn_model = Sequential()
# create embedding layer — usually the first layer in text problems
# vocabulary size — number of unique words in data
##rnn_model.add(Embedding(input_dim = VOCABULARY_SIZE, 
### length of vector with which each word is represented
## output_dim = EMBEDDING_SIZE, 
### length of input sequence
## input_length = MAX_SEQ_LENGTH, 
### False — don’t update the embeddings
## trainable = False 
##))
# add an RNN layer which contains 64 RNN cells
# True — return whole sequence; False — return single output of the end of the sequence
rnn_model.add(SimpleRNN(64, 
 return_sequences=True
))
# add time distributed (output at each sequence) layer
rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
#compile model
rnn_model.compile(loss      =  'categorical_crossentropy',
                  optimizer =  'adam',
                  metrics   =  ['acc'])
# check summary of the model
##rnn_model.summary()

rnn_training = rnn_model.fit(X_train, y_train, batch_size=32, epochs=10,
                             validation_data=(X_test, y_test))
'''
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """
    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'
    # plot model loss
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('loss')
    ax1.tick_params('y')
    ax1.legend(loc='upper right', shadow=False)
    ax1.set_title('Model loss through #epochs', color=orange, fontweight='bold')
    # plot model accuracy
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='training')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='validation')
    ax2.set_xlabel('# epoch')
    ax2.set_ylabel('accuracy')
    ax2.tick_params('y')
    ax2.legend(loc='lower right', shadow=False)
    ax2.set_title('Model accuracy through #epochs', color=orange, fontweight='bold')
    plt.show()


model_ = 'LSTM'
if model_=='simpleRNN':
# ----- Define RNN model -----
    model = Sequential()
    model.add(SimpleRNN(128))
    model.add(Dense(MAX_SEQ_LENGTH, activation = 'sigmoid'))
    ##rnn_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
    # ----- Compile model -----
    ##rnn_model.compile(loss='mean_squared_error categorical_crossentropy binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4))

    rnn_model.compile(loss      =  'binary_crossentropy',
                      optimizer =  'adam',
                      metrics   =  ['acc'])


elif model_=='LSTM':
    # create architecture
    model = Sequential()
##    bidirect_model.add(Embedding(input_dim = VOCABULARY_SIZE,
##     output_dim = EMBEDDING_SIZE,
##     input_length = MAX_SEQ_LENGTH,
##     weights = [embedding_weights],
##     trainable = True
##    ))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dense(MAX_SEQ_LENGTH, activation = 'sigmoid'))
##    bidirect_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
    #compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    # check summary of model
##    model.summary()



    
# ----- Train model -----

hist = model.fit(x=X_train, y=y_train, batch_size=1,epochs=100,
                        validation_data=(X_test, y_test))


loss, accuracy = model.evaluate(X_test, y_test, verbose = 1)
print('Loss: {0},\nAccuracy: {1}'.format(loss, accuracy))

z = model.predict(X_test)
zz = model.predict(X_train)

plot_model_performance(
    train_loss=hist.history.get('loss', []),
    train_acc=hist.history.get('acc', []),
    train_val_loss=hist.history.get('val_loss', []),
    train_val_acc=hist.history.get('val_acc', [])
)


def my_metric(y_true, y_pred):
    a = y_pred
    a[a > 0.5] = 1
    a[a <= 0.5] = 0
    pwei = precision_score(y_true, a, average='weighted')
    pmic = precision_score(y_true, a, average='micro')
    rwei = recall_score(y_true, a, average='weighted')
    rmic = recall_score(y_true, a, average='micro')
    fwei = f1_score(y_true, a, average='weighted')
    fmic = f1_score(y_true, a, average='micro')
    result = [pwei,pmic,rwei,rmic,fwei,fmic]
    return result

trainpred = model.predict(X_train)
testpred  = model.predict(X_test)
trainresults = my_metric(y_train, trainpred)
testresults = my_metric(y_test, testpred)
model.save(f'{model_}_model.h5')
print('trainresults:',trainresults)
print('testresults:',testresults)
np.save(f'{model_}_trainresults.npy',trainresults)
np.save(f'{model_}_testresults.npy',testresults)
