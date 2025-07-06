import numpy as np
from numpy import random
import scipy.sparse as sp

import argparse
def create_parser():
    parser = argparse.ArgumentParser()
    # 添加需要的参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--cuda', default=True, help='有无cuda')

    # 读取设置的参数
    args = parser.parse_args()
    print(args)

def ex_mat(n):  # 查看函数
    mat = np.identity(n, dtype='float32')
    print(mat)


def ex_map():  # 对指定的序列做映射
    cls = {'wang': 1, 'long': 2, 'tao': 3}
    # re = map(cls, )
    print('re')


def ex_random():
    random.seed(3)
    for i in range(5):
        # random.seed(3)
        print(random.random())


def ex_sp_csrmatrix():
    A = np.array([[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 6], [7, 0, 8, 9]])
    print(A)
    A = sp.csr_matrix(A)
    print(A)
    print(A.data, A.indices)
    print(A.indptr)
def ex_multiply():
    ads = np.array([[1,0,1],[0,0,1],[1,0,0]])
    ads_adj = ads + np.multiply(ads.T,(ads.T > ads)) - np.multiply(ads, (ads.T > ads))
    print(ads)
    print(ads_adj)

if __name__ == '__main__':
    # ex_mat(5)
    # ex_map()
    # ex_random()
    # ex_sp_csrmatrix()
    ex_multiply()
