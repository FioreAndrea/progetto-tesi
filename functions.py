import constant as c
import pandas as pd
import os
from os.path import join
from itertools import combinations
import scipy as sp
from scipy.spatial.distance import euclidean, hamming, cityblock
import numpy as np


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
            if type == 'euclidean':
                D[int(i), int(j)] = euclidean(p, c)
            if type == 'hamming':
                D[int(i), int(j)] = hamming(p, c)
            if type == 'manhattan':
                D[int(i), int(j)] = cityblock(p, c)
            if type == 'chi':
                D[int(i), int(j)] = chi2_distance(p, c)

    return D