from collections import Counter
from math import log
import pandas as pd
import string

def glue_tokens(tokens, order):
    return '{0}@{1}'.format(order, ' '.join(tokens))

def unglue_tokens(tokenstring, order):
    if order == 1:
        return [tokenstring.split("@")[1].replace(" ","")]
    return tokenstring.split("@")[1].split(" ")

def tokenize_sentence(sentence, order):
    sentence = sentence.lower()
    sentence = ((sentence.replace('Ô¨Å', '*').replace('Ô¨Ç', '*')).replace('Äö', "'")).replace('Ô¨Ç', '*')
    sentence = ((((sentence.strip(string.punctuation)).replace(",", "")).replace("'", "")).replace('.', '')).replace(
        '...', '')
    sentence = sentence.replace('?', '')
    tokens = sentence.split()
    tokens = ['<s>'] * (order - 1) + tokens + ['</s>']
    return tokens

def unigram(corpus, label, target):
    unigrams = Counter()
    for sent, lab in zip(corpus, label):
        if isinstance(sent, str) == True:
            if lab == target:
                words = tokenize_sentence(sent, 1)
                # print(words)
                # print("tokenized", words)
                for w in words:
                    unigrams[w] += 1
    unigram_total = sum(unigrams.values())
    frequent_vocab = []
    for w, v in unigrams.items():
        if v >= 2:
            frequent_vocab.append(w)

    context = []
    uni = Counter()
    for sent, lab in zip(corpus, label):
        if isinstance(sent, str) == True:
            if lab == target:
                words = tokenize_sentence(sent, 1)
                for w in words:
                    if w not in frequent_vocab:
                        word = '<unk/>'
                    else:
                        word = w
                    context.append(word)
                    uni[word] += 1
    total = sum(uni.values())
    return uni, total

class ent_perp_frame():
    def train(corpus, label, target, type):

        if type == 'gender' or type == 'character':
            uni, total = unigram(corpus, label, target)
            s = 0
            N = 0
            ent = []
            perp = []
            for sent in corpus:
                if isinstance(sent, str) == True:
                    words = tokenize_sentence(sent, 1)  # tokenize sentence with the order 1 as the parameter
                    sent_s = 0  # recording non-normalized entropy for this sentence
                    sent_N = 0  # total number of words in this sentence (for normalization)
                    for w in words:
                        prob = uni[w] / total
                        if prob == 0:
                            prob = 1 / total
                        logprob = log(prob, 2)  # the log of the prob to base 2
                        s += -log(prob, 2)  # add the neg log prob to s
                        sent_s += -log(prob, 2)  # add the neg log prob to sent_s
                        N += 1  # increment the number of total words
                        sent_N += 1  # increment the number of total words in this sentence
                    sent_cross_entropy = sent_s / sent_N
                    sent_perplexity = 2 ** sent_cross_entropy
                    # print(words, "cross entropy:", sent_cross_entropy, "perplexity:", sent_perplexity)
                    ent.append(sent_cross_entropy)
                    perp.append(sent_perplexity)
                else:
                    ent.append(0)
                    perp.append(0)
            frame = pd.DataFrame({'{}_ent'.format(target): pd.Series(ent)})#,  ent
                                  #'{}_perp'.format(target): pd.Series(perp)}) ######FIX ME BACK
        return frame, uni

    def test(corpus, label, uni, target, type):
        total = sum(uni.values())
        if type == 'gender' or type == 'character':
            uni, total = unigram(corpus, label, target)
            s = 0
            N = 0
            ent = []
            perp = []
            for sent in corpus:
                if isinstance(sent, str) == True:
                    words = tokenize_sentence(sent, 1)  # tokenize sentence with the order 1 as the parameter
                    sent_s = 0  # recording non-normalized entropy for this sentence
                    sent_N = 0  # total number of words in this sentence (for normalization)
                    for w in words:
                        prob = uni[w] / total
                        if prob == 0:
                            prob = 1 / total
                        logprob = log(prob, 2)  # the log of the prob to base 2
                        s += -log(prob, 2)  # add the neg log prob to s
                        sent_s += -log(prob, 2)  # add the neg log prob to sent_s
                        N += 1  # increment the number of total words
                        sent_N += 1  # increment the number of total words in this sentence
                    sent_cross_entropy = sent_s / sent_N
                    sent_perplexity = 2 ** sent_cross_entropy
                    # print(words, "cross entropy:", sent_cross_entropy, "perplexity:", sent_perplexity)
                    ent.append(sent_cross_entropy)
                    perp.append(sent_perplexity)
                else:
                    ent.append(0)
                    perp.append(0)
            frame = pd.DataFrame({'{}_ent'.format(target): pd.Series(ent)})#,  ent
                                  #'{}_perp'.format(target): pd.Series(perp)}) FIX ME ALSO
        return frame

