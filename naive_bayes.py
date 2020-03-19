from collections import Counter
from glob import glob
import numpy as np
import pickle
from collections import defaultdict
from nltk.sentiment import util
from tqdm import tqdm
import re
import time

def make_unigram_feature_set(documents, min_freq=1, mark_negation=False):
    """
    This function goes through a corpus and retains all candidate unigram features
     making a feature set. Optionally, it can also preprocess the corpus annotating
     with _NEG words that are in the scope of a negation (using NLTK helper functions).
    
    :param documents: all documents, each a list of words
    :param min_freq: minimum frequency of a token for it to be part of the feature set
    :param mark_negation: whether to preprocess the document using NLTK's nltk.sentiment.util.mark_negation
        see the documentation `nltk.sentiment.util.mark_negation?`
    :returns: unigram feature set
    """
    
    counter = Counter()
    for doc in documents:
        if mark_negation == True:
            doc = util.mark_negation(doc)
        counter.update(doc)
    features = []
    for f, n in counter.most_common():
        if n >= min_freq:
            features.append(f)
        else:
            break
    return frozenset(features)

def inspect_set(input_set, k=5, neg=False):
    """
    Helper function to inspect a few elements in a set of features
    
    :param input_set: a set of features
    :param k: how many elements to select
    :param neg: return `*_NEG` features only
    :returns: up to k elements 
    """
    selected = set()
    for w in input_set:
        if len(selected) < k:            
            if not neg:
                selected.add(w)
            elif '_NEG' in w:
                selected.add(w)
        else:
            break
    return selected


def make_feature_map(document, feature_set, 
                     binary=True, 
                     mark_negation=False):
    """
    This function takes a document, possibly pre-processes it by marking words in the scope of negation, 
     and constructs a dict indicating which features in `feature_set` fire. Features may be binary, 
     flagging occurrence, or integer, indicating the number of occurrences.
     If no feature can be extracted, a special feature is fired, namely 'EMPTY()'.
     
    :param document: a list of words
    :param feature_set: set of features we are looking for
    :param binary: whether we are indicating presence or counting features in feature_set
    :param mark_negation: whether we should apply NLTK's mark_negation to document before applying the feature function
    :returns: dict with entries 'contains(f)=1/0' for binary features or 'count(f)=n' for count features
    """
    if mark_negation:
        document = util.mark_negation(document)
    dic = defaultdict(float)
    for i in feature_set:
        if i in document:
            if binary:
                x = f"contains({i})"
                dic[x] = 1.0
            else:
                x = f"count({i})"
                if x not in dic:
                    dic[x] = 1.0
                else:
                    dic[x] += 1.0
    if len(dic) == 0:
        dic["EMPTY()"] = 1.0
    return dic

def make_cpd(raw_counts, alpha, v):
    """
    This converts a dictionary of raw counts into a cpd.

    :param raw_counts: dict where a key is a feature and a value is its counts (without pseudo counts)
        this should already include the 'EMPTY()' feature
    :param alpha: how many pseudo counts should we add per observation
    :param v: the size of the feature set (already including the 'EMPTY()' feature)
    :returns: a cpd as a dict where a key is a feature and a value is its smoothed probability
    """
    total = sum(raw_counts.values())
    for i in raw_counts:
        raw_counts[i] = (raw_counts[i] + alpha) / (alpha * v)
    return raw_counts
        

class NaiveBayesClassifier:
    
    def __init__(self, training_pos, training_neg, binary, mark_negation, alpha=0.1, min_freq=2):
        """
        :param training_pos: positive documents
            a document is a list of tokens
        :param training_neg: negative documents
            a document is a list of tokens
        :param binary: whether we are using binary or count features
        :param mark_negation: whether we are pre-processing words in negation scope
        :param alpha: Laplace smooth pseudo count
        :param min_freq: minimum frequency of a token for it to be considered a feature
        """
                
        # Make feature set
        print('Extracting features:')
        feature_set = make_unigram_feature_set(
            training_pos + training_neg,  # we use a concatenation of positive and negative training instances
            min_freq=min_freq, 
            mark_negation=mark_negation)
        
        print(' %d features' % len(feature_set))
        
        # Estimate model: 1/2) count        
        print('MLE: counting')        
        counts = [defaultdict(float), defaultdict(float)]
        for docs, y in [(training_pos, 1), (training_neg, 0)]:
            for doc in tqdm(docs):  # for each document
                # we extract features
                fmap = make_feature_map(doc, 
                                        feature_set, 
                                        binary=binary, 
                                        mark_negation=mark_negation)
                # and gather counts for the pair (y, f)
                for f, n in fmap.items():
                    counts[y][f] += n  
                                
        # 2/2) Laplace-1 MLE
        #  we put EMPTY() is in the support
        print('MLE: smoothing')
        counts[0]['EMPTY()'] += 0
        counts[1]['EMPTY()'] += 0
        # and compute cpds using Laplace smoothing
        self._cpds = [
            make_cpd(counts[0], alpha, len(feature_set) + 1),  # we add 1 because we want EMPTY() to add towards total
            make_cpd(counts[1], alpha, len(feature_set) + 1)]
        print('MLE: done')
        # Store data
        self._binary = binary
        self._mark_negation = mark_negation
        self._alpha = alpha
        self._feature_set = feature_set
            
    def get_log_parameter(self, f, y):
        """Returns log P(f|y)"""
        return np.log(self._cpds[y].get(f, self._cpds[y]['EMPTY()']))
        
    def classify(self, doc):
        """
        This function classifies a document by extracting features <f_1...f_n> for it 
         and then computing 
            log P(<f_1...f_n>|Y=0) and log P(<f_1...f_n>|Y=1)
         and finally picking the best (that is, either Y=0 or Y=1).
        
        :param doc: a list of tokens
        :returns: 0 or 1 (the argmax of log P(<f_1...f_n>|y))
        """
        doc = make_feature_map(doc, 
                        self._feature_set, 
                        binary=self._binary, 
                        mark_negation=self._mark_negation)
        c_0 = 0
        c_1 = 0
        for i in doc:
            c_0 += self.get_log_parameter(i, 0)
            c_1 += self.get_log_parameter(i, 1)
#         print(c_0, c_1)
        return np.argmax([c_0, c_1])
    

def evaluate_model(classifier, pos_docs, neg_docs):
    """
    :param classifier: an NaiveBayesClassifier object
    :param pos_docs: positive documents
    :param neg_docs: negative documents
    :returns: a dictionary containing the number of
        * true positives
        * true negatives
        * false positives
        * false negatives
     as well as 
        * accuracy
        * precision
        * recall 
        * and [F1](https://en.wikipedia.org/wiki/F1_score)
    """
    dic = {"TP" : 0, "TN" : 0, "FP" : 0, "FN" : 0, "A" : 0, "P" : 0, "R" : 0, "F1" : 0}
    
    # Calculate counts for 'False negatives' and 'True positives'
    for i in pos_docs:
        result = classifier.classify(i)
        if result == 0:
            dic["FN"] += 1
        else:
            dic["TP"] += 1
            
    # Calculate counts for 'True negatives' and 'False positives'
    for i in neg_docs:
        result = classifier.classify(i)
        if result == 0:
            dic["TN"] += 1
        else:
            dic["FP"] += 1
            
    # Calculate the 'Recall', 'Precision', 'Accuracy' and 'F1-score'
    dic["R"] = dic["TP"] / (dic["TP"] + dic["FN"])
    dic["P"] = dic["TP"] / (dic["TP"] + dic["FP"])
    dic["A"] = (dic["TP"] + dic["TN"]) / (dic["TP"] + dic["TN"] + dic["FN"] + dic["FP"])
    dic["F1"] = 2 * (dic["P"] * dic["R"]) / (dic["P"] + dic["R"])
    
    return dic
    

train_or_test = input("train or test? ")

if train_or_test == "train":
    start_time = time.time()
    train_pos_mask = 'aclImdb/train/pos/*.txt'
    train_neg_mask = 'aclImdb/train/neg/*.txt'
    pos = glob(train_pos_mask)
    neg = glob(train_neg_mask)
    pos_train_set = []
    neg_train_set = []
    for i in tqdm(pos):
        with open(i) as f:
            for line in f:
                line = re.sub(r'[^\w\s]','',line)
                line = line.lower().split(' ')
                pos_train_set.append(line)
    for i in tqdm(neg):
        with open(i) as f:
            for line in f:
                line = re.sub(r'[^\w\s]','',line)
                line = line.lower().split(' ')
                neg_train_set.append(line)

    test_pos_mask = 'aclImdb/test/pos/*.txt'
    test_neg_mask = 'aclImdb/test/neg/*.txt'
    pos = glob(test_pos_mask)
    neg = glob(test_neg_mask)
    pos_test_set = []
    neg_test_set = []
    for i in tqdm(pos):
        with open(i) as f:
            for line in f:
                line = re.sub(r'[^\w\s]','',line)
                line = line.lower().split(' ')
                pos_test_set.append(line)
    for i in tqdm(neg):
        with open(i) as f:
            for line in f:
                line = re.sub(r'[^\w\s]','',line)
                line = line.lower().split(' ')
                neg_test_set.append(line)

    pickle.dump(pos_test_set, open("pos_test_set_naive_bayes", 'wb'))
    pickle.dump(neg_test_set, open("neg_test_set_naive_bayes", 'wb'))
    pickle.dump(pos_train_set, open("pos_train_set_naive_bayes", 'wb'))
    pickle.dump(neg_train_set, open("neg_train_set_naive_bayes", 'wb'))

    classifier = NaiveBayesClassifier(
        pos_train_set, neg_train_set, 
        binary=True, mark_negation=False,
        alpha=1, min_freq=2)

    metrics = evaluate_model(classifier, pos_test_set, neg_test_set)
    pickle.dump(metrics, open("metrics_naive_bayes", 'wb'))    
    pickle.dump(classifier, open("classifier_naive_bayes", 'wb'))

    print('Development')
    print('TP %d TN %d FP %d FN %d' % (metrics['TP'], metrics['TN'], metrics['FP'], metrics['FN']))
    print('P %.4f R %.4f A %.4f F1 %.4f' % (metrics['P'], metrics['R'], metrics['A'], metrics['F1']))

    print("--- %s seconds ---" % (time.time() - start_time))
elif train_or_test == "test":
    start_time = time.time()
    pos_train_set = pickle.load(open("pos_train_set_naive_bayes", 'rb'))
    neg_train_set = pickle.load(open("neg_train_set_naive_bayes", 'rb'))
    training_docs = pos_train_set + neg_train_set
    pos_test_set = pickle.load(open("pos_test_set_naive_bayes", 'rb'))
    neg_test_set = pickle.load(open("neg_test_set_naive_bayes", 'rb'))
    classifier = pickle.load(open("classifier_naive_bayes", 'rb'))
    dev_metrics = pickle.load(open("dev_metrics_naive_bayes", 'rb'))

    #ADD EXTRA EVAL OPTIONS HERE
    print('Development')
    print('TP %d TN %d FP %d FN %d' % (dev_metrics['TP'], dev_metrics['TN'], dev_metrics['FP'], dev_metrics['FN']))
    print('P %.4f R %.4f A %.4f F1 %.4f' % (dev_metrics['P'], dev_metrics['R'], dev_metrics['A'], dev_metrics['F1']))
    print("--- %s seconds ---" % (time.time() - start_time))
else:
    print("Error occured")
