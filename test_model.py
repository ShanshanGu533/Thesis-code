import matplotlib as mpl
mpl.use("Agg")
import os
from sys import argv
from random import seed
import warnings

import numpy as np
from numpy import logspace, log10
from sklearn.linear_model import SGDClassifier


    ##########
    # set up #
    ##########
    seed(44)
    np.random.seed(44)


    seqs_weight = [0.9, 1.1, 1.1, 0.9]
    bin_mtx = np.array([[1, 0, 0, 0], [0, 0,0, 1], [0, 1, 0, 0], [0, 0, 0,1]])
    col = np.array([1, 18, 19, 2])
    ALPHA_RANGE = list(logspace(-3, log10(1), 15)) + [10, 20, 30, 40]
    print(ALPHA_RANGE)
    for alpha in ALPHA_RANGE:
        clf = SGDClassifier(loss='log', penalty='elasticnet',
                            alpha=alpha, l1_ratio=0.99,
                            n_jobs=2, max_iter=10000,
                            random_state=33, tol=1e-3)
        clf.fit(bin_mtx, col, sample_weight= seqs_weight)
        print(clf.predict([[1, 0, 0, 0]]))
