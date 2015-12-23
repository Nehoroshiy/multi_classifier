"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import random
import numpy as np

import matplotlib.pyplot as plt

from dataset.data_utils import get_cifar10_data
from classifiers import Softmax

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
    results = {}
    best_val = -1
    best_softmax = None
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [5e4, 1e8]


    def accuracy(ethalon, pred):
        return np.average(ethalon == pred)
    ################################################################################
    # TODO:                                                                        #
    # Use the validation set to set the learning rate and regularization strength. #
    # This should be identical to the validation that you did for the SVM; save    #
    # the best trained softmax classifer in best_softmax.                          #
    ################################################################################
    best_val = 0.
    for rate in np.linspace(learning_rates[0], learning_rates[1], 3):
        for reg_str in np.linspace(regularization_strengths[0], regularization_strengths[1], 3):
            print('rate: {}, reg: {}'.format(rate, reg_str))
            sm = Softmax()
            sm.train(X_train, y_train, rate, reg_str)
            pred_train, pred_test = None, None
            pred_train = sm.predict(X_train)
            pred_val = sm.predict(X_val)
            if best_val < accuracy(y_val, pred_val):
                best_val = accuracy(y_val, pred_val)
            results[(rate, reg_str)] = (accuracy(y_train, pred_train), accuracy(y_val, pred_val))

    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy)

    print 'best validation accuracy achieved during cross-validation: %f' % best_val
