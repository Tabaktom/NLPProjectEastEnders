from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.utils import shuffle


def trainClassifier(trainData_X, trainData_Y, l, type):
    if type == 'gender':
        fitted = LogisticRegression(solver = 'newton-cg', C = l).fit(trainData_X, trainData_Y) #1 no reg, 0 full reg
    elif type == 'character':
        fitted = LogisticRegression(solver = 'saga', multi_class = 'multinomial', C = l).fit(trainData_X, trainData_Y) #1 no reg, 0 full reg
    return fitted

def predictLabels(reviewSamples, classifier):
    return classifier.predict(reviewSamples)

def crossValidate(Xdataset, Ydataset, lam, folds, type):
    #dataset = pd.concat([Xdataset, Ydataset], axis=1)
    average_results_array = []
    foldsize = int(len(Xdataset)/folds)
    for l in lam:
        results = []
        for i in range(0, len(Xdataset), int(foldsize)):
            print('Fold start on items {} - {}, using Lambda = {}'.format(i, i+foldsize, l))
            myTestData_X = Xdataset[i:i+foldsize]
            myTestData_y = Ydataset[i:i+foldsize]
            myTrainData_X = pd.concat([Xdataset[:i],  Xdataset[i+foldsize:]], axis = 0)
            myTrainData_Y = pd.concat([Ydataset[:i], Ydataset[i + foldsize:]], axis = 0)
            classifier = trainClassifier(myTrainData_X, myTrainData_Y, l, type)
            y_pred = predictLabels(myTestData_X, classifier)
            results.append(precision_recall_fscore_support(myTestData_y, y_pred, average = 'weighted'))
        precision = []
        recall = []
        fscore = []
        for res in results:
            precision.append(res[0])
            recall.append(res[1])
            fscore.append(res[2])
        avgresults = [sum(precision)/len(precision), sum(recall)/len(recall), sum(fscore)/len(fscore)]
        average_results_array.append(avgresults)
    return average_results_array[fscore.index(max(fscore))], lam[fscore.index(max(fscore))]