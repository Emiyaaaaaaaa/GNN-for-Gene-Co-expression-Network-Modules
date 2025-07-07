import numpy as np
import scipy.sparse as sp
import torch
import sys

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # 类必须在编码之前排序，才能启用静态类编码.
    # In other words, make sure the first class always maps to index 0.
    # 换句话说，确保第一个类始终映射到索引0.
    classes = sorted(list(set(labels)))  # 去重+排序 7
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 字典：one-hot
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)  # map 映射函数
    return labels_onehot

def normalize_adj(mx):  # 归一化
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()  # 计算 np.power(x, y) 计算 x 的 y 次方  flatten(dim)从dim展开
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.  # 将正无穷的值设置为 0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)  # sp.diags 生成对角矩阵 (2708, 2708)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)  # transpose() <==> T

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 行求和
    r_inv = np.power(rowsum, -1).flatten()  # rowsum的-1次方 再flatten变行
    r_inv[np.isinf(r_inv)] = 0. # 无穷置零
    r_mat_inv = sp.diags(r_inv) # 对角化
    mx = r_mat_inv.dot(mx)  # 乘积方式 归一化操作
    return mx

def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 将所有零保存在计算机内存中的效率很低，更合适的方法是只保存非零元素以及位置信息
    labels = encode_onehot(idx_features_labels[:, -1])  # 2708个标签进行ONE-HOT编码
    # sys.getsizeof(features)
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}  # 生成真正的字典数据顺序标签 2708 个序号
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)  # 5429 对边
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape) # 5429 重新编号
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),dtype=np.float32)
    # 制定数据  生成矩阵 5429 不对称的 邻接矩阵     sp.coo_matrix(data, (row,col)) '坐标格式'
    # build symmetric adjacency matrix   建立对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # multiply 对位相乘

    features = normalize_features(features)  # 归一化函数 (2708, 1433)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))  # 添加自环操作

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))  # matrix.todense()将稀疏矩阵转为稠密矩阵
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])  # ...[0, 0, 1, ..., 0, 0, 0] -->tensor([2, 5, 4,  ..., 1, 0, 2])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    adj, features, labels, idx_train, idx_val, idx_test = load_data()