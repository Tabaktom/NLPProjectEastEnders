import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

import numpy as np
import string
from collections import OrderedDict

import re
import string


df = pd.read_csv('training.csv')
df.columns = ['Line', 'Char', 'Gen']

training = df.Line[ : int(len(df)*0.8)]
heldout =  df.Line[int(len(df)*0.8):]
char_y_training = df.Char[: int(len(df)*0.8)]
char_y_hedlout = df.Char[int(len(df)*0.8):]
gen_y_training = df.Gen[: int(len(df)*0.8)]
gen_y_heldout = df.Gen[int(len(df)*0.8):]


def Punctuation(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string


class baseline():
    def gender(y_vector):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for y in y_vector:
            rand = np.random.randint(low=0, high=2, size=1)
            if rand == 1:
                pred = 'male'
                if pred == y:
                    TP += 1
                else:
                    FP += 1
            else:
                pred = 'female'
                if pred == y:
                    TN += 1
                else:
                    FN += 1
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        Fscore = 2 * (recall * precision) / (recall + precision)
        return Fscore

    def characters(y_vector, characters_list):
        # CHECK THIS ACCURACY
        correct = 0
        incorrect = 0
        predictions = []
        for ind, y in enumerate(y_vector):
            rand = np.random.randint(low=0, high=len(characters), size=1)[0]
            char = characters_list[rand]
            predictions.append(char)
        predictions = pd.Series(predictions)
        fscore = precision_recall_fscore_support(y_vector, predictions, average='weighted')
        report = classification_report(y_vector, predictions)
        return fscore
    # fscore[2]

###SIMPLE FEATURE EXTRACTION###


characters = list(set(char_y_training))
for index, char in enumerate(characters):
    characters[index] = characters[index][0] + characters[index][1:].lower()

class feature_extract():
    def simple_features(training):
        line_length = []
        questions = []
        exclamations = []
        ellipsis = []
        cutoff = []
        gossip = []
        sentences_per_line = []
        words_per_sent = []
        emphasis = []
        for index, row in enumerate(training):
            gossip_count = 0
            typ = isinstance(row, str)
            if typ == True:
                row = ((row.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
                sentences = sent_tokenize(row)
                s_cnt = 0
                w_cnt = 0
                for s in sentences:
                    s_cnt += 1
                    words = s.split(' ')
                    for ind in range(len(words)):
                        words[ind] = Punctuation(words[ind])
                    w_cnt += len(words)
                emphasis.append(row.count('*') / 2)
                sentences_per_line.append(s_cnt)
                words_per_sent.append(w_cnt / s_cnt)
                questions.append(row.count('?'))
                exclamations.append(row.count('!'))
                ellipsis.append(row.count('...'))
                cutoff.append(row.count('--'))
                line_length.append(len(word_tokenize(row)))
                for char in characters:
                    if char in row:
                        gossip_count += 1
                gossip.append(gossip_count)
            else:
                questions.append(0)
                exclamations.append(0)
                ellipsis.append(0)
                line_length.append(0)
                cutoff.append(0)
                gossip.append(0)
                words_per_sent.append(0)
                sentences_per_line.append(0)
                emphasis.append(0)
        simple_features = pd.DataFrame({'questions': pd.Series(questions),
                                        'exclamations': pd.Series(exclamations),
                                        'cutoff': pd.Series(cutoff),
                                        'ellipsis': pd.Series(ellipsis),
                                        'line_length': pd.Series(line_length),
                                        'gossip': pd.Series(gossip),
                                        'send_words': pd.Series(words_per_sent),
                                        'sent_in_line': pd.Series(sentences_per_line),
                                        'emphasis': pd.Series(emphasis)})
        return simple_features

    ######################################################################################
    def bigram_frame(training):
        bigram_final = OrderedDict()
        bigram_vocab = []
        for index, row in enumerate(training):
            typ = isinstance(row, str)
            if typ == True:
                row = ((row.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
                sentences = sent_tokenize(row)
                for s in sentences:
                    words = s.split(' ')
                    for ind in range(len(words)):
                        words[ind] = Punctuation(words[ind])
                    bigram = ngrams(words, 2)
                    for gg in bigram:
                        gg = str((str(gg).strip("')(',")).replace(",", "")).replace("'", "")
                        # print(gg)
                        if gg not in bigram_vocab:
                            bigram_vocab.append(gg)
                            bigram_final[gg] = 1
                        else:
                            bigram_final[gg] += 1
        return bigram_final

    ######################################################################################
    def unigram_frame(training):
        unigram_final = OrderedDict()
        unigram_vocab = []
        for index, row in enumerate(training):
            if isinstance(row, str) == True:
                row = ((row.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
                sentences = sent_tokenize(row)
                for s in sentences:
                    s = (((s.strip(string.punctuation)).replace(",", "")).replace("'", "")).split(' ')
                    for word in s:
                        # word = lemmatizer.lemmatize(word.lower())
                        word = (word.strip(string.punctuation).replace(",", "")).lower()
                        if '...' in word:
                            word = word.split('...')
                            for w in word:
                                if w not in unigram_vocab:
                                    unigram_vocab.append(w)
                                    unigram_final[w] = 1
                                else:
                                    unigram_final[w] += 1
                        elif '.' in word:
                            word = word.split('.')
                            for w in word:
                                if w not in unigram_vocab:
                                    unigram_vocab.append(w)
                                    unigram_final[w] = 1
                                else:
                                    unigram_final[w] += 1
                        else:
                            if word not in unigram_vocab:
                                unigram_vocab.append(word)
                                unigram_final[word] = 1
                            else:
                                unigram_final[word] += 1
        return unigram_final

    ######################################################################################
    def bigram_df_generation(training):
        bigram_final = feature_extract.bigram_frame(training)
        bigram_frame = {}
        bigram_vocab = []
        for bigram in list(bigram_final.keys()):
            bigram_frame[bigram] = []
        counting = 0
        for index, row in enumerate(training):
            if isinstance(row, str) == True:
                row = ((row.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
                row = (((row.strip(string.punctuation)).replace(",", "")).replace("'", "")).split(' ')
                bigram = ngrams(row, 2)
                bi_list = []
                for ind, x in enumerate(bigram):
                    bi_list.append(
                        str((((str(x).strip("')(',")).replace(",", "")).replace("'", "")).replace(".", "")).replace("?",
                                                                                                                    ""))
                    # print(gg)
                for bi in list(bigram_final.keys()):
                    if bi in bi_list:
                        bigram_frame[bi].append(1)
                        counting += 1
                    else:
                        bigram_frame[bi].append(0)
            else:
                for bi in list(bigram_final.keys()):
                    bigram_frame[bi].append(0)
        bigram_dataframe = pd.DataFrame(bigram_frame)
        return bigram_dataframe

    ######################################################################################
    def unigram_df_generation(training):
        unigram_final = feature_extract.unigram_frame(training)
        ordered = feature_extract.narrow_ngram(unigram_final, min_count=0, max_count=500)
        frame = {}
        for word in list(ordered.values()):
            frame[word] = []
        for index, row in enumerate(training):
            if isinstance(row, str) == True:
                # row = ((row.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
                line = ((row.strip(string.punctuation)).replace(",", "")).replace("'", "")
                for word in list(ordered.values()):
                    if word in line:
                        frame[word].append(1)
                    else:
                        frame[word].append(0)
            else:
                for word in list(ordered.values()):
                    frame[word].append(0)

        arrays = list(frame.keys())
        final_frame = {}
        for words, a in zip(list(frame.keys()), list(frame.values())):
            final_frame[words] = pd.Series(a)
        unigram_features = pd.DataFrame(final_frame)
        return unigram_features

    ######################################################################################
    def narrow_ngram(unigram_final, min_count, max_count):
        if isinstance(unigram_final, dict) == False:
            unigram_final = str(unigram).strip("][")
            print(unigram_final)
            print(type(unigram_final))
        values = list(unigram_final.values())
        keys = list(unigram_final.keys())
        val_array = []
        key_array = []
        for ind in range(len(keys)):
            if values[ind] >= min_count and values[ind] <= max_count:
                key_array.append(keys[ind])
                val_array.append(values[ind])
        ordered = OrderedDict()
        for v, k in zip(val_array, key_array):
            ordered[v] = k
        return ordered

simple_features = feature_extract.simple_features(training)
unigram_dataframe = feature_extract.unigram_df_generation(training)
bigram_dataframe = feature_extract.bigram_df_generation(training)
massive_cols = pd.concat([simple_features, unigram_dataframe, bigram_dataframe], axis = 1, sort = False)

###Character Classification###
X_train, X_test, y_train, y_test = train_test_split(massive_cols, char_y_training, test_size = 0.33, shuffle = True)
print('--------------')
print('Character')
print('--------------')
###Logsitic###
print('Logistic...')
clf = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial').fit(X_train, y_train)
y_hat_train = clf.predict(X_train)
train_fscore = precision_recall_fscore_support(y_train, y_hat_train, average = 'weighted')
y_hat_test = clf.predict(X_test)
char_class_lr = classification_report(y_hat_test, y_test)
print(char_class_lr)
test_fscore = precision_recall_fscore_support(y_test, y_hat_test,average = 'weighted')
#print('Precision || Recall || FScore || Support')
#print('The Training Accuracy: {}'.format(train_fscore))
#print('The Test Accuracy: {}'.format(test_fscore))

###Naive Bayes
print('Naive Bayes...')
model = GaussianNB().fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
char_class_nb = classification_report(y_hat_test, y_test)
print(char_class_nb)
train_fscore = precision_recall_fscore_support(y_train, y_hat_train, average = 'weighted')
test_fscore = precision_recall_fscore_support(y_test, y_hat_test, average = 'weighted')
#print('Train Accuracy: {} and Test Accuracy: {}'.format(train_fscore, test_fscore))

###Gender Classification###
print('------------------')
print('Gender')
print('------------------')
#Logistic
print('Logistic...')
X_train, X_test, y_train, y_test = train_test_split(simple_features, gen_y_training, test_size = 0.33, shuffle = True)
clf = LogisticRegression().fit(X_train, y_train)

y_hat_train = clf.predict(X_train)
y_hat_test = clf.predict(X_test)

gen_class_lr = classification_report(y_hat_test, y_test)
print(gen_class_lr)
train_fscore = precision_recall_fscore_support(y_train, y_hat_train, average = 'weighted')
test_fscore = precision_recall_fscore_support(y_test, y_hat_test, average = 'weighted')

#print('The Training Accuracy: {}'.format(train_fscore))
#print('The Test Accuracy: {}'.format(test_fscore))

###Naive Bayes
print('Naive Bayes...')
model = GaussianNB().fit(X_train, y_train)
y_hat_train = model.predict(X_train)
y_hat_test = model.predict(X_test)
gen_class_nb = classification_report(y_hat_test, y_test)
print(gen_class_nb)
train_fscore = precision_recall_fscore_support(y_train, y_hat_train, average = 'weighted')
test_fscore = precision_recall_fscore_support(y_test, y_hat_test, average = 'weighted')
print('Train Accuracy: {} and Test Accuracy: {}'.format(train_fscore, test_fscore))









