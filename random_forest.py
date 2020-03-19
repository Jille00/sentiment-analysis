from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.tree import plot_tree
import numpy as np
import glob
import re
from tqdm import tqdm
import pickle
import time

neg_train = glob.glob("aclImdb/train/neg/*.txt")
pos_train = glob.glob("aclImdb/train/pos/*.txt")
neg_test = glob.glob("aclImdb/test/neg/*.txt")
pos_test = glob.glob("aclImdb/test/pos/*.txt")



def classify(corpus, classes, clf, vectorizer):
    classify = []
    for sentence in tqdm(corpus):
        classify.append(clf.predict(vectorizer.transform([sentence]).toarray()))
    
    return classify

train_or_test = input("train or test? ")
if train_or_test == "train":
    start_time = time.time()
    sentences = []
    classes = []
    for neg, pos in tqdm(zip(neg_train, pos_train)):
        with open(neg) as f:
            line1 = f.read().lower()
            line1 = re.sub(r'[^\w\s]','',line1)
            sentences.append(line1)
            classes.append(0)
        with open(pos) as f:
            line2 = f.read().lower()
            line2 = re.sub(r'[^\w\s]','',line2)
            sentences.append(line2)
            classes.append(1)

    Y = np.array(classes)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(sentences)
    clf = RandomForestClassifier()
    clf.fit(X, Y)

    pickle.dump(X, open("X_train_random_forest", 'wb'))
    pickle.dump(Y, open("Y_train_random_forest", 'wb'))
    pickle.dump(clf, open("Model_random_forest", 'wb'))

    corpus = []
    classes2 = []

    for file in tqdm(neg_test):
        with open(file) as f:
            line = f.read().lower()
            line = re.sub(r'[^\w\s]','',line)
            corpus.append(line)
            classes2.append(0)

    for file in tqdm(pos_test):
        with open(file) as f:
            line = f.read().lower()
            line = re.sub(r'[^\w\s]','',line)
            corpus.append(line)
            classes2.append(1)

    classify = classify(corpus, classes2, clf, vectorizer)

    count = 0    
    for n, value in enumerate(classes2):
        if classify[n] == classes2[n]:
            count +=1

    print("Accuracy: ",(count/len(classes2))*100) 

    print("--- %s seconds ---" % (time.time() - start_time))

elif train_or_test == "test":
    start_time = time.time()
    clf = pickle.load(open("Model_random_forest", 'rb'))
    
    sentences = []
    classes = []
    for neg, pos in tqdm(zip(neg_train, pos_train)):
        with open(neg) as f:
            line1 = f.read().lower()
            line1 = re.sub(r'[^\w\s]','',line1)
            sentences.append(line1)
            classes.append(0)
        with open(pos) as f:
            line2 = f.read().lower()
            line2 = re.sub(r'[^\w\s]','',line2)
            sentences.append(line2)
            classes.append(1)

    corpus = []
    classes2 = []

    for file in tqdm(neg_test):
        with open(file) as f:
            line = f.read().lower()
            line = re.sub(r'[^\w\s]','',line)
            corpus.append(line)
            classes2.append(0)

    for file in tqdm(pos_test):
        with open(file) as f:
            line = f.read().lower()
            line = re.sub(r'[^\w\s]','',line)
            corpus.append(line)
            classes2.append(1)

    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit_transform(sentences)
    classify = classify(corpus, classes2, clf, vectorizer)

    count = 0    
    for n, value in enumerate(classes2):
        if classify[n] == classes2[n]:
            count +=1

    print("Accuracy: ", (count/len(classes2))*100)   
    print("--- %s seconds ---" % (time.time() - start_time))


else:
    print("An error occured")
