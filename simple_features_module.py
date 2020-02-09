import string
from nltk import sent_tokenize, word_tokenize
import pandas as pd

def Punctuation(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string

class feature_extract():
    def simple_features(training, characters):
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
                    s_cnt +=1
                    words = s.split(' ')
                    for ind in range(len(words)):
                        words[ind] = Punctuation(words[ind])
                    w_cnt += len(words)
                emphasis.append(row.count('*')/2)
                sentences_per_line.append(s_cnt)
                words_per_sent.append(w_cnt/s_cnt)
                questions.append(row.count('?'))
                exclamations.append(row.count('!'))
                ellipsis.append(row.count('...'))
                cutoff.append(row.count('--'))
                line_length.append(len(word_tokenize(row)))
                for char in characters:
                    if char in row:
                        gossip_count +=1
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
                                        'cutoff':pd.Series(cutoff),
                                        'ellipsis': pd.Series(ellipsis),
                                        'line_length': pd.Series(line_length),
                                        'gossip':pd.Series(gossip),
                                        'sent_words':pd.Series(words_per_sent),
                                        'sent_in_line': pd.Series(sentences_per_line),
                                        'emphasis':pd.Series(emphasis)})
        return simple_features