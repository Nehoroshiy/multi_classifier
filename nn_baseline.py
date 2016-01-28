"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
import matplotlib.pyplot as plt
from classifier_trainer import ClassifierTrainer
from gradient_check import eval_numerical_gradient
from classifiers.neural_net_baseline import *
from dataset import get_cifar10_data



def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_cifar10_data()
    print 'Train data shape: ', X_train.shape
    print 'Train labels shape: ', y_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Validation labels shape: ', y_val.shape
    print 'Test data shape: ', X_test.shape
    print 'Test labels shape: ', y_test.shape

    image_shape = (3, 32, 32)
    classes = 10
    N = 100
    hidden_size = 100
    input_size = int(np.prod(image_shape))

    model = init_two_layer_model(input_size, hidden_size, classes)

    X = np.random.randn(N, *image_shape)
    X = X.reshape((N, -1))
    y = np.random.randint(classes, size=N)

    loss, _ = two_layer_net(X, model, y, reg=0)

    # Sanity check: Loss should be about log(10) = 2.3026
    print 'Sanity check loss (no regularization): ', loss

    # Sanity check: Loss should go up when you add regularization
    loss, _ = two_layer_net(X, model, y, reg=1)
    print 'Sanity check loss (with regularization): ', loss

    X_train = X_train.T
    X_val = X_val.T

    hidden_size = 1000
    model = init_two_layer_model(X_train.shape[1], hidden_size, classes)
    trainer = ClassifierTrainer()
    best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
            X_train[:50], y_train[:50], X_val, y_val, model, two_layer_net,
            reg=0.001, update='momentum', momentum=0.9, learning_rate=0.0001, batch_size=10, num_epochs=10,
            verbose=True)


    hidden_size = 1000

    model = init_two_layer_model(X_train.shape[1], hidden_size, classes)
    trainer = ClassifierTrainer()
    best_model, loss_history, train_acc_history, val_acc_history = trainer.train(
              X_train, y_train, X_val, y_val, model, two_layer_net,
              reg=0.001, momentum=0.9, learning_rate=0.0001, batch_size=50, num_epochs=1,
              acc_frequency=50, verbose=True)