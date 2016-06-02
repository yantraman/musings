import numpy as np

import keras

from keras.layers.convolutional import Convolution1D, MaxPooling1D  # for text data we have 1D convolution
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence

from random import random_sample

np.random.seed(1337)  # for reproducibility

# TODO
# change all prints to python logging
# add comments and details about the code

hyper_para = list()

hyper_para.append((1e-4, 1e-2, 'real'))
hyper_para.append((10, 1000, 'int'))
hyper_para.append((1, 9, 'int'))
hyper_para.append((50, 500, 'int'))
hyper_para.append((1e-1, 9e-1, 'real'))

hyper_values = list()
hyper_values.append([0.001, 100, 3, 100, 0.25])

for i in range(0):
    hyper_values.append(random_sample(hyper_para))

res = list()  # to store the results

for i in range(len(hyper_values)):
    values = hyper_values[i]
    print values
    max_features = 5000
    maxlen = 10
    batch_size = 8
    embedding_dims = 100
    nb_filter = 100
    filter_length = 3
    hidden_dims = 100
    nb_epoch = 3

    learningRate = values[0]
    hidden_dims = values[1]
    filter_length = values[2]
    numberOfFilters = values[3]
    overfitting = values[4]

    print 'Loading tweet data...'

    import utils

    ((X_train, y_train), (X_test, y_test), ind2word) = \
        utils.load_data(M=max_features, is_ngram=True,
                        rtype='index')

    print (len(X_train), 'train sequences')
    print (len(X_test), 'test sequences')

    print 'Pad sequences'
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print 'shuffle the training example'
    arr = np.arange(len(X_train))
    np.random.shuffle(arr)
    X_train = X_train[arr]
    y_train = np.array(y_train)[arr]
    y_train = y_train.tolist()

    arr = np.arange(len(X_test))
    np.random.shuffle(arr)
    X_test = X_test[arr]
    y_test = np.array(y_test)[arr]
    y_test = y_test.tolist()

    print 'Build model...'
    model = Sequential()

    model.add(Embedding(max_features, embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(overfitting))

    model.add(Convolution1D(nb_filter=numberOfFilters,
                            filter_length=filter_length, border_mode='valid',
                            activation='relu', subsample_length=1))

    model.add(MaxPooling1D(pool_length=2))

    model.add(Flatten())

    model.add(Dense(hidden_dims))
    model.add(Dropout(overfitting))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    opt = keras.optimizers.RMSprop(lr=learningRate)
    model.compile(loss='binary_crossentropy', optimizer=opt)

    nTest = len(X_test)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test[:nTest / 2], y_test[:nTest / 2]))
    score, acc = model.evaluate(X_test[nTest / 2:], y_test[nTest / 2:], batch_size=batch_size)

    print ('Test score:', score)
    print ('Test accuracy:', acc)

    res.append(acc)

ind = 0
prev = 0
for i in range(len(hyper_values)):
    if res[i] > prev:
        prev = res[i]
        ind = i
optimal_parameters = hyper_values[ind]
