import constant as c
import pandas as pd
import os
from os.path import join
from itertools import combinations
import scipy as sp
from scipy.spatial.distance import euclidean, hamming, cityblock
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from DS3 import DS3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler
import proxmin



def chi2_distance(A, B):
    # compute the chi-squared distance using above formula
    chi = 0.5 * np.nansum([((a - b) ** 2) / (a + b)
                           for (a, b) in zip(A, B)])
    return chi


def create_D(current, previous, features, type):
    D = np.empty([len(previous), len(current)])
    for i in range(len(previous)):
        p = previous.loc[i, features]
        # print(p)
        for j in range(len(current)):
            c = current.loc[j, features]
            # print(c)
            if type == 'e':
                D[int(i), int(j)] = euclidean(p, c)
            if type == 'h':
                D[int(i), int(j)] = hamming(p, c)
            if type == 'm':
                D[int(i), int(j)] = cityblock(p, c)
            if type == 'c':
                D[int(i), int(j)] = chi2_distance(p, c)

    return D


def runDS3(D, reg, verbose = False):
    """
            This function runs DS3.

            :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
            :param p: norm to be used to calculate regularization cost.
            :returns: regularization cost.
    """
    # initialize DS3 class with dis-similarity matrix and the regularization parameter.
    dis_matrix = D
    reg = 0.5
    DS = DS3(dis_matrix, reg)
    # run the ADMM(p=inf) algorithm.
    start = time.time()
    data_admm, num_of_rep_admm, obj_func_value_admm, obj_func_value_post_proc_admm, Z = \
        DS.ADMM(mu=10 ** -1, epsilon=10 ** -7, max_iter=200000, p=np.inf)
    end = time.time()
    rep_super_frames = data_admm

    # change the above indices into 0s and 1s for all indices.
    N = len(D)
    summary = np.zeros(N)
    for i in range(len(rep_super_frames)):
        summary[rep_super_frames[i]] = 1

    run_time = end - start
    obj_func_value = obj_func_value_admm
    idx = []
    for index, i in enumerate(summary):
        if i == 1:
            idx.append(index)

    idx = np.asarray(idx)
    if verbose:
        print('Object function value :', obj_func_value)
        print("Run Time :", run_time)
        print("Objective Function Value  :", obj_func_value)
        print("Summary :", summary)
        print("Index representative :", idx)

    return idx


def run_logisticRegression(previous, current, idx, ds3 = False,verbose = False, plot = False):
    """
        This function trains and uses model.

        :param D: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.
        :returns: regularization cost.
    """

    training_D3 = previous.iloc[idx].reset_index()
    train = previous
    scaler = StandardScaler()
    if ds3:
        scaler.fit(training_D3[c.features])
        X_train = scaler.transform(training_D3[c.features])
        y_train = training_D3['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']
    else:
        scaler.fit(train[c.features])
        X_train = scaler.transform(train[c.features])
        y_train = train['bug']
        X_test = scaler.transform(current[c.features])
        y_test = current['bug']

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    ##PREDICT##
    y_predicted = clf.predict(X_test)
    ##CALCULATE SCORE OF THE MODEL##
    score = clf.score(X_test, y_test)
    if True:
        print(f'- LogisticRegression score: {score}')
    # CONFUCIO MATRIX##
    cm = metrics.confusion_matrix(y_test, y_predicted)
    ##PLOT CONFUSION MATRIX##
    if plot:
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = 'Confusion matrix \n Accuracy Score: {0}'.format(score)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title(all_sample_title, size=15);
    cmm = metrics.multilabel_confusion_matrix(y_test, y_predicted)
    if verbose:
        print("Confusion multiclass matrix :\n", cmm)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_predicted))
        print("Accuracy:", metrics.accuracy_score(y_test, y_predicted))
        print("Precision:", metrics.precision_score(y_test, y_predicted, average='weighted'))
        print("Recall:", metrics.recall_score(y_test, y_predicted, average='weighted'))

    recall = metrics.recall_score(y_test, y_predicted, average='weighted')
    precision = metrics.precision_score(y_test, y_predicted, average='weighted')

    gmean = geometric_mean_score(y_test, y_predicted, average='macro')
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)

    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    balance = 1 - np.sqrt(((0 - FPR) ** 2 + (1 - TPR) ** 2) / 2)
    balance = np.average(balance)

    ##F MEASURE##
    fmeasure = f1_score(y_test, y_predicted, average='macro')

    if verbose:
        #print('TPR :', TPR)
        #print('FPR :', FPR)
        print('F-Measure : ', fmeasure)
        print('G-Mean :', gmean)
        print('Balance :', balance)
