import cPickle as pickle
import numpy as np
import sys
import os
import gc

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        del datadict['data']
        del datadict['labels']
        del datadict
        #gc.collect()
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    np_names = ['np_train_data', 'np_train_labels', 'np_test_data', 'np_test_labels']
    print(all([os.path.isfile(os.path.join(ROOT, name)) for name in np_names]))
    if all([os.path.isfile(os.path.join(ROOT, name)) for name in np_names]):
        print("Np files were found")
        # load np arrays from given files
        xt, yt, xte, yte = [np.fromfile(os.path.join(ROOT, name), dtype=np.uint8) for name in np_names]
        xt = xt.reshape(-1, 32, 32, 3).astype("float")
        xte = xte.reshape(-1, 32, 32, 3).astype("float")
        print(xt.shape, yt.shape, xte.shape, yte.shape)
        return xt, yt, xte, yte
    else:
        xs = []
        ys = []
        for b in range(1,6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        size_estimator = sum(x.shape[0] for x in xs)
        x_all = np.zeros((size_estimator,) + xs[0].shape[1:], dtype=float)
        y_all = np.zeros(size_estimator, dtype=int)
        cur_size = 0
        for x, y in zip(xs, ys):
            x_all[cur_size:cur_size + x.shape[0]] = x
            y_all[cur_size:cur_size + x.shape[0]] = y
            cur_size += x.shape[0]
            del x
            del y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        for arr, fname in zip([x_all, y_all, Xte, Yte], np_names):
            arr.astype(np.uint8).tofile(os.path.join(ROOT, fname))
        return x_all, y_all, Xte, Yte