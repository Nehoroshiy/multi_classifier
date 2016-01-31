"""
2015-2016 Constantine Belev const.belev@ya.ru
"""

import numpy as np
import scipy as sp

import scipy.misc

from matplotlib import pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from manopt.approximator import ImageApproximator


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def generate_sigma_set(shape, percent):
    return sp.sparse.random(*shape, density=percent).nonzero()


def rearrange(matrix, shape1, shape2):
    block_view = matrix.T.reshape((shape1[1], shape2[1], shape1[0], shape2[0]))
    rearranged_block_view = block_view.transpose((0, 2, 1, 3))
    rearranged_matrix = rearranged_block_view.reshape(shape1[0]*shape1[1], shape2[0] * shape2[1])
    return rearranged_matrix


def invert_rearrange(rearranged_matrix, shape1, shape2):
    rearranged_block_view = rearranged_matrix.reshape(shape1[::-1] + shape2[::-1])
    block_view = rearranged_block_view.transpose((0, 2, 1, 3))
    matrix = block_view.reshape((shape1[1] * shape2[1], shape1[0] * shape2[0])).T
    return matrix


def rearrange_sparse(matrix, shape1, shape2):
    matrix.tocoo()
    idx, idy = matrix.nonzero()
    i_outer, j_outer = idy // shape2[1], idy % shape2[1]
    i_inner, j_inner = idx // shape2[0], idx % shape2[0]
    idx_rearranged, idy_rearranged = i_outer * shape1[0] + i_inner, j_outer * shape2[0] + j_inner
    return coo_matrix((matrix.data.copy(), (idx_rearranged, idy_rearranged)), shape=(shape1[0] * shape1[1], shape2[0] * shape2[1]))


def invert_rearrange_sparse(rearranged_matrix, shape1, shape2):
    rearranged_matrix.tocoo()
    idx_rearranged, idy_rearranged = rearranged_matrix.nonzero()
    block_height, all_height = shape1
    block_width, all_width = shape2

    idx = (idx_rearranged % block_height) * block_width + (idy_rearranged % block_width)
    idy = (idx_rearranged // block_height) * all_width + (idy_rearranged // block_width)
    return coo_matrix((rearranged_matrix.data.copy(), (idx, idy)), shape=(np.multiply(shape1, shape2)))


def plot_gray_image(image):
    plt.gray()
    plt.imshow(image)
    plt.show()


def ordinary_approximation_old(a, sigma_set, rmax=None, os=8, maxiter=700):
    rmax = int(len(sigma_set[0]) / ((os + 1) * np.sum(a.shape)))

    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort(kind='mergesort')]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort(kind='mergesort')]

    results = []
    val_errors = []
    history_images = np.zeros((rmax - 1,) + a.shape)
    for r in range(1, rmax):
        train_percent = 1. * os * r * np.sum(a.shape) / len(sigma_set[0])
        print('approximation for rank {}, percent {}:'.format(r, train_percent))
        train_mask = np.array(np.random.binomial(1., train_percent, len(sigma_set[0])), dtype=np.bool)
        train_set = (sigma_set[0][train_mask], sigma_set[1][train_mask])
        validation_set = (sigma_set[0][~train_mask], sigma_set[1][~train_mask])

        train_data = a[train_set]
        validation_data = a[validation_set]
        a_sparse = csr_matrix(coo_matrix((train_data, train_set), shape=a.shape))
        approximator = CGApproximator()
        #x_prev = results[-1][0] if results else None
        x_prev = None
        results.append(approximator.approximate(a_sparse, r, train_set, x0=x_prev, maxiter=maxiter, eps=1e-10))
        a_validation = csr_matrix(coo_matrix((validation_data, validation_set), shape=a.shape))
        val_errors.append(np.average(rel_error(results[-1][0].evaluate(validation_set).data, validation_data)))
        print('error reached: {}. validation error: {}'.format(results[-1][-1][-1], val_errors[-1]))
        #plot_gray_image(results[-1][0].full_matrix())
        history_images[r - 1, ...] = results[-1][0].full_matrix()
    history_images.tofile('history_riemannian')
    np.array(val_errors).tofile('val_errors_riemannian')
    print('best model has rank:{} with val_error = {}'.format(np.argmin(val_errors) + 1, min(val_errors)))
    best_model = results[np.argmin(val_errors)]
    return best_model


def emergency_approximation(a, sigma_set, maxiter=700):
    rmax=min((5,) + a.shape)
    train_percent = 0.9
    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort(kind='mergesort')]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort(kind='mergesort')]


    train_mask = np.array(np.random.binomial(1., train_percent, len(sigma_set[0])), dtype=np.bool)
    train_set = (sigma_set[0][train_mask], sigma_set[1][train_mask])
    validation_set = (sigma_set[0][~train_mask], sigma_set[1][~train_mask])
    validation_data = np.array(a[validation_set]).ravel()

    results = []
    val_errors = []
    for r in range(1, rmax + 1):
        print('approximation for rank {}/{}:'.format(r, rmax))
        train_data = np.array(a[train_set]).ravel()
        a_sparse = csr_matrix(coo_matrix((train_data, train_set), shape=a.shape))
        approximator = CGApproximator()
        x_prev = None
        results.append(approximator.approximate(a_sparse, r, train_set, x0=x_prev, maxiter=maxiter, eps=1e-10))
        a_validation = csr_matrix(coo_matrix((validation_data, validation_set), shape=a.shape))
        val_errors.append(np.linalg.norm(results[-1][0].evaluate(validation_set).data - validation_data))
        print('error reached: {}. validation error: {}'.format(results[-1][-1][-1], val_errors[-1]))
    np.array(val_errors).tofile('val_errors_kron')
    print('best model has rank:{} with val_error = {}'.format(np.argmin(val_errors) + 1, min(val_errors)))
    best_model = results[np.argmin(val_errors)]
    return best_model



def ordinary_approximation(a, sigma_set, rmax=None, os=4, maxiter=700):
    rmax = int(1.0 * len(sigma_set[0]) / ((os + 0.5) * np.sum(a.shape)))
    if rmax == 0:
        return emergency_approximation(a, sigma_set, maxiter=700)

    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort(kind='mergesort')]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort(kind='mergesort')]

    train_all_percent = 1. * os * rmax * np.sum(a.shape) / ((os + 0.5) * rmax * np.sum(a.shape))
    train_all_mask = np.array(np.random.binomial(1., train_all_percent, len(sigma_set[0])), dtype=np.bool)
    train_all_set = (sigma_set[0][train_all_mask], sigma_set[1][train_all_mask])
    validation_set = (sigma_set[0][~train_all_mask], sigma_set[1][~train_all_mask])
    validation_data = np.array(a[validation_set]).ravel()

    results = []
    val_errors = []
    #history_images = np.zeros((rmax,) + a.shape)
    for r in range(1, rmax + 1):
        train_percent = 1. * r / rmax
        print('approximation for rank {}/{}, train percent {}:'.format(r, rmax, train_percent))
        train_mask = np.array(np.random.binomial(1., train_percent, len(train_all_set[0])), dtype=np.bool)
        train_set = (train_all_set[0][train_mask], train_all_set[1][train_mask])

        train_data = np.array(a[train_set]).ravel()
        a_sparse = csr_matrix(coo_matrix((train_data, train_set), shape=a.shape))
        approximator = ImageApproximator()
        #x_prev = results[-1][0] if results else None
        x_prev = None
        results.append(approximator.approximate(a_sparse, r, train_set, x0=x_prev, maxiter=maxiter, eps=1e-10))
        a_validation = csr_matrix(coo_matrix((validation_data, validation_set), shape=a.shape))
        val_errors.append(np.linalg.norm(results[-1][0].evaluate(validation_set).data - validation_data))
        print('error reached: {}. validation error: {}'.format(results[-1][-1][-1], val_errors[-1]))
        #plot_gray_image(invert_rearrange(results[-1][0].full_matrix(), (64, 32), (8, 16)))
        #history_images[r - 1, ...] = results[-1][0].full_matrix()
    #history_images.tofile('history_kron')
    np.array(val_errors).tofile('val_errors_kron')
    print('best model has rank:{} with val_error = {}'.format(np.argmin(val_errors) + 1, min(val_errors)))
    best_model = results[np.argmin(val_errors)]
    return best_model


def kron_approximation(a, sigma_set, shape1, shape2, rmax=None, os=5, maxiter=700):
    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort(kind='mergesort')]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort(kind='mergesort')]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort(kind='mergesort')]

    data = np.array(a[sigma_set]).ravel()
    a_sparse = csr_matrix(coo_matrix((data, sigma_set), shape=a.shape))
    a_rearranged = rearrange_sparse(a_sparse, shape1, shape2).tocsr()
    new_sigma_set = a_rearranged.nonzero()
    x, it, err = ordinary_approximation(a_rearranged, new_sigma_set)
    return invert_rearrange(x.full_matrix(), shape1, shape2)


def generate_kronecker_images(image, shape2, filename='kronecker_image'):
    shape1 = int(image.shape[0] / shape2[0]), int(image.shape[1]/ shape2[1])
    for delta_x in range(shape2[0]):
        for delta_y in range(shape2[1]):
            pass


"""
if __name__ == '__main__':
    lena = sp.misc.lena()
    m, n = lena.shape
    #mh, nh = int(m / 4), int(n / 4)
    #lena_part = lena[mh:2*mh, nh:2*nh].copy()
    lena_part = lena
    percent = 0.1
    sigma_set = generate_sigma_set(lena_part.shape, percent)

    lena_cutted = np.zeros_like(lena)
    lena_cutted[sigma_set] = lena[sigma_set]
    #plot_gray_image(lena_cutted)


    #building_blocks = [(8, 8), (8, 16), (16, 8), (16, 16)]
    #building_blocks = [(1, 2), (2, 1), (2, 4), (4, 2)]
    building_blocks = [(8, 8)]
    shapes = list(map(lambda x: ((int(lena.shape[0] / x[0]), int(lena.shape[1]/ x[1])), x), building_blocks))
    approxs = []
    counts = np.zeros_like(lena, dtype=int)
    for shape1, shape2 in shapes:
        for delta_x in range(shape2[0]): # range(0, shape2[0], max(int(shape2[0] / 4), 1)):
            for delta_y in range(shape2[1]): #range(0, shape2[1], max(int(shape2[1] / 4), 1)):
                x = np.zeros_like(lena)
                delta_arr = np.array([delta_x, delta_y], dtype=int)
                delta_size = np.array((x.shape - delta_arr) / shape2, dtype=int)
                #delta_size = int((x.shape[1] - delta_y) / shape2[1])
                max_size = delta_arr + delta_size * shape2
                #max_size = delta_y + delta_size * shape2[1]
                x_view = x[delta_arr[0]:max_size[0], delta_arr[1]:max_size[1]]
                #x_view = x[:, delta_y:max_size]
                mask = ((sigma_set[1] - delta_arr[1]) >= 0) & (sigma_set[1] < max_size[1])
                submask = ((sigma_set[0] - delta_arr[0]) >= 0) & (sigma_set[0] < max_size[0])
                new_sigma = (sigma_set[0][mask & submask] - delta_x, sigma_set[1][mask & submask] - delta_y)
                subshape = (x_view.shape[0] / shape2[0], x_view.shape[1] / shape2[1])
                counts[delta_arr[0]:max_size[0], delta_arr[1]:max_size[1]] += 1
                x_view[:, :] = kron_approximation(lena_cutted[delta_arr[0]:max_size[0]:, delta_arr[1]:max_size[1]].copy(), new_sigma, subshape, shape2)
                approxs.append(x)
        #approxs.append(kron_approximation(lena_cutted, sigma_set, shape1, shape2))
    img_buffer = np.stack(approxs)
    img_buffer.tofile('img_buffer_blocks')
    [plot_gray_image(image) for image in approxs]
    plot_gray_image(sum(approxs) / len(approxs))


    #x, it, err = ordinary_approximation(lena_part, sigma_set)
    plot_gray_image(lena_cutted)
    #plot_gray_image(x.full_matrix())

"""


def test():
    lena = sp.misc.lena()
    m, n = lena.shape
    percent = 0.3
    sigma_set = generate_sigma_set(lena.shape, percent)

    lena_cutted = np.zeros_like(lena)
    lena_cutted[sigma_set] = lena[sigma_set]

    building_blocks = [(8, 8)]
    shapes = map(lambda x: (tuple(np.asarray(lena.shape, dtype=int) / x), x), building_blocks)
    approxs = []
    for shape1, shape2 in shapes:
        x = kron_approximation(lena_cutted, sigma_set, shape1, shape2)
    return x


if __name__ == '__main__':
    x = test()
    plot_gray_image(x)
    print(x.min(), x.max())