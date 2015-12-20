"""
Copyright (c) 2015-2016 Constantine Belev



Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import fileinput
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy as sp
from pandas.io.common import ZipFile
from scipy import sparse

from manopt.sparse.approx.gd import cg

if sys.version_info[0] < 3:
    pass
else:
    pass

from io import BytesIO

def get_movielens_data(local_file=None):
    if not local_file:
        print('Downloading data...')
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        zip_response = get(zip_file_url)
        zip_content = BytesIO(zip_response.content)
        fname = 'ml-10M100K/ratings.dat'
        with ZipFile(zip_content) as zfile:
            zfile.extract(fname)
        for line in fileinput.input([fname], inplace=True):
            print(line.replace('::', ';'))
    else:
        fname = local_file
    ml_data = pd.read_csv(fname, sep=';', header=None, engine='c',
                                  names=['userid', 'movieid', 'rating', 'timestamp'],
                                  usecols=['userid', 'movieid', 'rating'])

    # normalize indices to avoid gaps
    ml_data['movieid'] = ml_data.groupby('movieid', sort=False).grouper.group_info[0]
    ml_data['userid'] = ml_data.groupby('userid', sort=False).grouper.group_info[0]

    # build sparse user-movie matrix
    data_shape = ml_data[['userid', 'movieid']].max() + 1
    data_matrix = sp.sparse.csr_matrix((ml_data['rating'],
                                       (ml_data['userid'], ml_data['movieid'])),
                                        shape=data_shape, dtype=np.float64)
    print('Done.')
    return data_matrix

def split_data(data, test_ratio=0.2):
    '''Randomly splits data into training and testing datasets. Default ratio is 80%/20%.
    Returns datasets in namedtuple format for convenience. Usage:
    train_data, test_data = split_data(data_matrix)
    or
    movielens_data = split_data(data_matrix)
    and later in code:
    do smth with movielens_data.train
    do smth with movielens_data.test
    '''

    num_users = data.shape[0]
    idx = np.zeros((num_users,), dtype=bool)
    sel = np.random.choice(num_users, int(test_ratio*num_users), replace=False)
    np.put(idx, sel, True)

    Movielens_data = namedtuple('MovieLens10M', ['train', 'test'])
    movielens_data = Movielens_data(train=data[~idx, :], test=data[idx, :])
    return movielens_data


if __name__ == "__main__":
    local_file_name = 'ml-10M100K/ratings.dat'
    data_matrix = get_movielens_data(local_file_name)
    train_matrix, test_matrix = split_data(data_matrix, test_ratio=0.995)

    print('Matrix build of shape {} step done. Start sorting sigma_set'.format(train_matrix.shape))

    # sorted sigma set
    sigma_set = train_matrix.nonzero()
    sigma_set[0][:] = sigma_set[0][sigma_set[1].argsort()]
    sigma_set[1][:] = sigma_set[1][sigma_set[1].argsort()]
    sigma_set[1][:] = sigma_set[1][sigma_set[0].argsort()]
    sigma_set[0][:] = sigma_set[0][sigma_set[0].argsort()]

    print('sigma_set sorting done. Start iterations')

    x = None
    maxiter_ordinary = 10
    r = 6
    for rank in range(1, r):
        current_maxiter = 1 #np.log(rank) + 1
        x, it, err = cg(train_matrix, sigma_set, rank, x0=x, maxiter=int(maxiter_ordinary * current_maxiter))
        #if rank in [3, 4, 5]:
        #    cProfile.run('cg(train_matrix, sigma_set, rank, x0=x, maxiter=maxiter_ordinary * current_maxiter)')
        print('Iterations for rank {} done with err {}'.format(rank, err[-1]))
        print('Current sigmas: {}'.format(x.s))
        if it != int(maxiter_ordinary * current_maxiter):
            r = rank
            break
    print('rank is {}'.format(r))
    print(x.s)
    x, it, err = cg(train_matrix, sigma_set, r=6, x0=x, maxiter=200)
