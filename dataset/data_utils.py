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


def get_cifar10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    import os
    print(os.curdir)
    cifar10_dir = 'dataset/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    mask = np.arange(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = np.arange(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = np.arange(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T

    return X_train, y_train, X_val, y_val, X_test, y_test