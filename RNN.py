from keras.preprocessing import sequence
from collections import Counter
from glob import glob
import numpy as np
import pickle
from collections import defaultdict
from nltk.sentiment import util
from tqdm import tqdm
import re
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from random import shuffle

def Punctuation(string): 
  
    # punctuation marks 
    punctuations = '''!()[]{};:'",<>./?@#$%^&*_~0123456789='''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in string.lower(): 
        if x in punctuations: 
            string = string.replace(x, " ") 
  
    # Print string without punctuation 
    return string 

indices= []
with open('imdb.vocab') as f:
    for line in f:
        indices.append(line.rstrip())

train_or_test = input("train or test? ")

if train_or_test == "train":
    train_mask = 'train/*.txt'
    train_pos_mask = 'train/pos/*.txt'
    train_neg_mask = 'train/neg/*.txt'
    pos = glob(train_pos_mask)
    neg = glob(train_neg_mask)
    pos_train_set = []
    neg_train_set = []
    for i in tqdm(pos):
        with open(i) as f:
            for line in f:
                line = Punctuation(line)
                line = line.lower().split(' ')
                li = []
                for j in line:
                    if j != '' and j != '-':
                        try:
                            li.append(indices.index(j))
                        except:
                            pass
                pos_train_set.append(li)
    for i in tqdm(neg):
        with open(i) as f:
            for line in f:
                line = Punctuation(line)
                line = line.lower().split(' ')
                li = []
                for j in line:
                    if j != '' and j != '-':
                        try:
                            li.append(indices.index(j))
                        except:
                            pass
                neg_train_set.append(li)


    train_mask = 'test/*.txt'
    test_pos_mask = 'test/pos/*.txt'
    test_neg_mask = 'test/neg/*.txt'
    pos = glob(test_pos_mask)
    neg = glob(test_neg_mask)
    pos_test_set = []
    neg_test_set = []
    for i in tqdm(pos):
        with open(i) as f:
            for line in f:
                line = Punctuation(line)
                line = line.lower().split(' ')
                li = []
                for j in line:
                    if j != '' and j != '-':
                        try:
                            li.append(indices.index(j))
                        except:
                            pass
                pos_test_set.append(li)
    for i in tqdm(neg):
        with open(i) as f:
            for line in f:
                line = Punctuation(line)
                line = line.lower().split(' ')
                li = []
                for j in line:
                    if j != '' and j != '-':
                        try:
                            li.append(indices.index(j))
                        except:
                            pass
                neg_test_set.append(li)


    pickle.dump(pos_test_set, open("pos_test_set", 'wb'))
    pickle.dump(neg_test_set, open("neg_test_set", 'wb'))
    pickle.dump(pos_train_set, open("pos_train_set", 'wb'))
    pickle.dump(neg_train_set, open("neg_train_set", 'wb'))

    training_docs = []
    for i in pos_train_set:
        training_docs.append((i, 1))
    for i in neg_train_set:
        training_docs.append((i, 0))
    shuffle(training_docs)

    testing_docs = []
    for i in pos_test_set:
        testing_docs.append((i, 1))
    for i in neg_test_set:
        testing_docs.append((i, 0))
    shuffle(testing_docs)

    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in training_docs:
        X_train.append(i[0])
        Y_train.append(i[1])

    for i in testing_docs:
        X_test.append(i[0])
        Y_test.append(i[1])

    pickle.dump(X_train, open("X_train", 'wb'))
    pickle.dump(X_test, open("X_test", 'wb'))
    pickle.dump(Y_train, open("Y_train", 'wb'))
    pickle.dump(Y_test, open("Y_test", 'wb'))

    max_words = 600
    X_train = sequence.pad_sequences(X_train, maxlen=max_words)
    X_test = sequence.pad_sequences(X_test, maxlen=max_words)

    embedding_size=32
    model=Sequential()
    model.add(Embedding(100000, embedding_size, input_length=max_words))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())

    model.compile(loss='binary_crossentropy', 
                 optimizer='adam', 
                 metrics=['accuracy'])

    batch_size = 64
    num_epochs = 3
    X_valid, Y_valid = X_train[:batch_size], Y_train[:batch_size]
    X_train2, Y_train2 = X_train[batch_size:], Y_train[batch_size:]
    model.fit(X_train2, Y_train2, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=num_epochs)

    pickle.dump(model, open("RNN-model", 'wb'))

    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test accuracy:', scores[1])

elif train_or_test == "test":
    X_train = pickle.load(open("X_train", 'rb'))
    X_test = pickle.load(open("X_test", 'rb'))
    Y_train = pickle.load(open("Y_train", 'rb'))
    Y_test = pickle.load(open("Y_test", 'rb'))
    model = pickle.load(open("RNN-model", 'rb'))

    #ADD EXTRA EVAL OPTIONS HERE
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print('Test accuracy:', scores[1])

else:
    print("An error occured")
