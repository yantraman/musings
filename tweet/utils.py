import numpy as np
import re
import os
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
from numpy.random import shuffle

stopWordList = stopwords.words("english")  # stopword list


# TODO
# change all prints to python logging
# add comments and details about the code
def replace_repetition(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


def get_vector(s):
    feature_vector = list()
    words = s.split()
    for w in words:

        w = replace_repetition(w)

        w = w.strip('\'"?,.')

        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)

        if w in stopWordList or val is None:
            continue
        else:
            feature_vector.append(w.lower())
    return feature_vector


def load_data(m=None, split_data=0.1, is_ngram=True, rtype='index', dataset='small'):
    if dataset == 'small':
        dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/data_tweet/small_data/'
    elif dataset == 'large':
        dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../data/data_tweet/large_data/'
    else:
        raise ValueError

    fp = open(dataset_path + 'positive_tweet.txt', 'r')
    pos = fp.readlines()
    fp.close()
    fp = open(dataset_path + 'negative_tweet.txt', 'r')
    neg = fp.readlines()
    fp.close()
    X = np.asarray(pos + neg)
    y = np.array([1 for i in range(len(pos))] + [0 for i in range(len(neg))])

    # data shuffling
    ind = range(len(X))
    shuffle(ind)
    X = X[ind]
    y = y[ind]

    nb_sample = len(X)
    x_training = X[:int(split_data * nb_sample)]
    y_training = y[:int(split_data * nb_sample)]

    x_testing = X[int(split_data * nb_sample):]
    y_testing = y[int(split_data * nb_sample):]

    counter1 = Counter()
    for t in x_training:
        counter1.update(t.strip().split())

    counter2 = Counter()
    counter3 = Counter()

    if is_ngram:

        for i in range(len(x_training)):
            t = x_training[i]
            word_list = t.strip().split()

            gram2 = ngrams(word_list, 2)
            two_gram = ['_'.join(g) for g in gram2]
            counter2.update(two_gram)
            t += ' '.join(two_gram)

            gram3 = ngrams(word_list, 3)
            three_gram = ['_'.join(g) for g in gram3]
            counter3.update(three_gram)
            t += ' '.join(three_gram)

            x_training[i] = t

        # update the testing data so that it include 2gram and 3gram
        for i in range(len(x_testing)):
            t = x_testing[i]
            word_list = t.strip().split()

            gram2 = ngrams(word_list, 2)
            two_gram = ['_'.join(g) for g in gram2]
            t += ' '.join(two_gram)

            gram3 = ngrams(word_list, 3)
            three_gram = ['_'.join(g) for g in gram3]
            t += ' '.join(three_gram)

            x_testing[i] = t

        if not m:
            m = len(counter1)
        else:
            M = min(m, len(counter1)) - 1
            m2 = int(m * 0.1)
            m3 = int(m * 0.02)
            m1 = m - m2 - m3
    else:
        if not m:
            m = len(counter1)
        else:
            m = min(m, len(counter1)) - 1
        m1 = m
        m2 = 0
        m3 = 0

    word2ind = {}
    ind2word = {}

    most_common = counter1.most_common(m1) + counter2.most_common(m2) + counter3.most_common(m3)

    assert len(most_common) == M
    for i in range(M):
        tu = most_common[i]  # tu is a 2-tuple, the first is the word, and the second is the frequency
        word2ind[tu[0]] = i + 1
        ind2word[i + 1] = tu[0]

    x_training_word_list = list()
    x_training_ind_list = list()
    for i in range(len(x_training)):
        list_word = [word for word in x_training[i].strip().split() if word in word2ind]
        list_ind = [word2ind[word] for word in list_word if word in word2ind]
        x_training_word_list.append(list_word)
        x_training_ind_list.append(list_ind)

        x_testing_word_list = list()
        x_testing_ind_list = list()
    for i in range(len(x_testing)):
        list_word = [word for word in x_testing[i].strip().split() if word in word2ind]
        list_ind = [word2ind[word] for word in list_word if word in word2ind]
        x_testing_word_list.append(list_word)
        x_testing_ind_list.append(list_ind)

    if rtype == 'word':
        return (x_training_word_list, y_training), (x_testing_word_list, y_testing), word2ind
    elif rtype == 'index':
        return (x_training_ind_list, y_training), (x_testing_ind_list, y_testing), ind2word
    else:
        print "argument value error, should be 'index' or 'word'"
        raise ValueError


#

if __name__ == '__main__':
    import time

    start = time.time()
    max_features = 5000
    load_data(m=max_features, is_ngram=True, rtype='index')
    tmp1 = load_data(is_ngram=False, m=1000)
    tmp2 = load_data(is_ngram=False, m=1000000)
    tmp1 = load_data(m=1000)
    tmp2 = load_data(m=1000000)
    end = time.time()

    print "elapsed time", end - start
