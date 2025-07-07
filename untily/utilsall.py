from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from data_gens import gens_map
import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

import random

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix using torch."""
    # 检查是否为稀疏矩阵，仅在是稀疏矩阵时转换为密集矩阵
    if adj.is_sparse:
        adj = adj.to_dense()

    # 计算每一行的和
    rowsum = torch.sum(adj, dim=1, keepdim=True)

    # 计算 D^(-0.5)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

    # 创建对角矩阵 D^(-0.5)
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt.squeeze())

    # 计算归一化后的邻接矩阵
    normalized_adj = torch.matmul(torch.matmul(adj, d_mat_inv_sqrt).transpose(-2, -1), d_mat_inv_sqrt)

    # 如果原始输入是稀疏矩阵，将结果转换回稀疏矩阵
    if adj.is_sparse:
        normalized_adj = normalized_adj.to_sparse()

    return normalized_adj

def load_data_(dataset_str):

    metabolite_data = pd.read_excel('datee/所有代谢物的定量.xlsx')
    data_keys = metabolite_data.keys()[1:]

    selected_keys = []
    for i in range(0, len(data_keys), 5):
        group = data_keys[i:i + 5].tolist()  # 将 pandas.Index 对象转换为列表
        if len(group) >= 3:
            selected = random.sample(group, 3)
            selected_keys.extend(selected)

    metabolite_features = metabolite_data[selected_keys]

    transcriptome_data = pd.read_excel('datee/转录组基因表达量.xlsx')

    transcriptome_features = transcriptome_data.drop(columns=['#ID'])
    pca = PCA(n_components=30)
    metabolite_features_pca = pca.fit_transform(metabolite_features)
    transcriptome_features_pca = pca.fit_transform(transcriptome_features)

    from sklearn.metrics.pairwise import cosine_similarity

    similarity_matrix = cosine_similarity(metabolite_features_pca, metabolite_features_pca)
    ###############################################################
    top_n = 30    # 为每个节点选择相似度最高的前 100 个节点
    top_nodes = []
    for row in similarity_matrix:        # 获取当前行中元素的索引，并按相似度降序排序
        sorted_indices = np.argsort(row)[::-1]        # 选择前 100 个节点的索引
        top_indices = sorted_indices[:top_n]
        top_nodes.append(top_indices)
    # 构建边
    edges = []
    for source_node, target_nodes in enumerate(top_nodes):
        for target_node in target_nodes:
            # 避免自环（如果需要可以保留自环，去掉这个条件判断）
            if source_node != target_node:
                edges.append([source_node, target_node])
                edges.append([target_node, source_node])
    ###############################################################
    # 将结果转换为 numpy 数组
    edges_metabolite = np.array(edges)

    similarity_matrix = cosine_similarity(transcriptome_features_pca, transcriptome_features_pca)

    top_n = 30  # 为每个节点选择相似度最高的前 100 个节点
    top_nodes = []
    for row in similarity_matrix:  # 获取当前行中元素的索引，并按相似度降序排序
        sorted_indices = np.argsort(row)[::-1]  # 选择前 100 个节点的索引
        top_indices = sorted_indices[:top_n]
        top_nodes.append(top_indices)
    # 构建边
    edges = []
    for source_node, target_nodes in enumerate(top_nodes):
        for target_node in target_nodes:
            if source_node != target_node:
                edges.append([source_node, target_node])
                edges.append([target_node, source_node])

    edges_transcriptome = np.array(edges)

    similarity_matrix = cosine_similarity(transcriptome_features_pca, metabolite_features_pca)

    top_n = 30  # 为每个节点选择相似度最高的前 100 个节点
    top_nodes = []
    for row in similarity_matrix:  # 获取当前行中元素的索引，并按相似度降序排序
        sorted_indices = np.argsort(row)[::-1]  # 选择前 100 个节点的索引
        top_indices = sorted_indices[:top_n]
        top_nodes.append(top_indices)
    # 构建边
    edges = []
    for source_node, target_nodes in enumerate(top_nodes):
        for target_node in target_nodes:
            if source_node != target_node:
                edges.append([source_node, target_node])
                edges.append([target_node, source_node])

    edges = np.array(edges)

    edges_metabolite = edges_metabolite + edges_transcriptome.max() + 1
    edges[:, 1] = edges[:, 1] + len(transcriptome_features_pca)

    # edges_all = np.concatenate([edges_transcriptome, edges_metabolite, edges], axis=0)
    edges_all = np.concatenate([edges], axis=0)

    feature_all = np.concatenate([transcriptome_features_pca, metabolite_features_pca], axis=0)

    relations = [torch.ones(len(edge))*idx for idx, edge in enumerate([edges_transcriptome, edges_metabolite, edges])]
    # PyG要求格式为 [2, edge_num]
    edge_index = edges_all.T

    values = np.ones(len(edges_all))

    adj_sp = torch.sparse_coo_tensor(indices=edge_index, values=torch.FloatTensor(values), size=[len(feature_all), len(feature_all)])

    adj_dense = adj_sp.to_dense()

    normalized_adj = normalize_adj(adj_dense)

    relations = torch.cat(relations)

    edge_index = torch.LongTensor(edge_index)
    feature_all = torch.FloatTensor(np.array(feature_all))
    relations = torch.tensor(relations, dtype=torch.long)

    metab = metabolite_data['ID'].values
    trans = transcriptome_data['#ID'].values

    org_index = np.concatenate([trans, metab])

    return edge_index, feature_all, relations, normalized_adj, org_index, len(transcriptome_features_pca)


def load_data():
    # 使用 read_csv 方法读取 CSV 文件
    metabolite_data = pd.read_csv('datee/fpkm.genename2.csv')

    data_keys = metabolite_data.keys()[1:-2]
    col_feature = metabolite_data[data_keys].values.T
    row_feature = metabolite_data[data_keys].values

    GeneName_list = metabolite_data['GeneName'].tolist()

    gene_row_map = {gene: idx for idx, gene in enumerate(GeneName_list)}
    gene_row_index = {idx: gene  for idx, gene in enumerate(GeneName_list)}
    gene_row_num = len(gene_row_map)

    gene_col_map = {gene: idx for idx, gene in enumerate(data_keys)}
    gene_col_index = {idx: gene for idx, gene in enumerate(data_keys)}
    gene_col_num = len(gene_col_map)


    gen_gen_edges = gens_map()
    gen_gen_edges = np.array(gen_gen_edges)[:, :-1]
    idx_1_map = np.array(list(map(gene_row_map.get, gen_gen_edges[:, 0])))
    idx_2_map = np.array(list(map(gene_row_map.get, gen_gen_edges[:, 1])))
    gen_gen_edges_map = np.dstack((idx_1_map, idx_2_map))[0]




    # 1. 将numpy数组转换为Python列表（便于遍历过滤）
    gen_gen_edges_list = gen_gen_edges_map.tolist()

    # 2. 过滤掉包含None的子列表（确保子列表中所有元素都不是None）
    filtered_list = [
        item for item in gen_gen_edges_list
        if item is not None  # 防止子列表本身是None
           and len(item) == 2  # 确保子列表有2个元素（符合原始结构）
           and all(x is not None for x in item)  # 子列表中两个元素都不是None
    ]
    # 3. 将过滤后的列表转换为int类型的numpy数组
    gen_gen_edges_map = np.array(filtered_list, dtype=int) + gene_col_num




    metabolite_features = col_feature
    transcriptome_features = row_feature

    pca = PCA(n_components=14)
    metabolite_features_pca = pca.fit_transform(metabolite_features)
    transcriptome_features_pca = pca.fit_transform(transcriptome_features)

    from sklearn.metrics.pairwise import cosine_similarity

    def batch_cosine_similarity(transcriptome_features, metabolite_features, batch_size=100, top_k=5):
        num_transcriptome = transcriptome_features.shape[0]
        num_metabolite = metabolite_features.shape[0]
        all_indices = []

        # 仅对transcriptome分批次，metabolite一次性加载（或全量计算）
        for i in tqdm(range(0, num_transcriptome, batch_size)):
            end_i = min(i + batch_size, num_transcriptome)
            batch_trans = transcriptome_features[i:end_i]  # 当前transcriptome批次

            # 计算当前批次与所有metabolite的相似度（全局计算）
            sim_matrix = cosine_similarity(batch_trans, metabolite_features)  # 形状：(batch_size, num_metabolite)

            # 对每个transcriptome节点，在全局metabolite中选top_k
            for row_idx in range(sim_matrix.shape[0]):
                # 全局排序，取top_k的索引
                top_indices = np.argsort(sim_matrix[row_idx])[-top_k:][::-1]  # 全局最相关的k个metabolite索引
                global_trans_idx = i + row_idx  # 转换为全局transcriptome索引

                # 构建边（transcriptome索引, metabolite索引）
                edges = np.array([[global_trans_idx, col_idx] for col_idx in top_indices])
                all_indices.append(edges)

        # 合并所有边
        return np.vstack(all_indices) if all_indices else np.array([])

    def batch_cosine_similarity_(transcriptome_features, metabolite_features, batch_size=100, top_k=5):
        num_transcriptome = transcriptome_features.shape[0]
        num_metabolite = metabolite_features.shape[0]
        indices = []

        for i in tqdm(range(0, num_transcriptome, batch_size)):
            end_i = min(i + batch_size, num_transcriptome)
            batch_transcriptome = transcriptome_features[i:end_i]

            for j in range(0, num_metabolite, batch_size):
                end_j = min(j + batch_size, num_metabolite)
                batch_metabolite = metabolite_features[j:end_j]

                # 计算当前批次的相似度
                sim_batch = cosine_similarity(batch_transcriptome, batch_metabolite)

                # 为每个节点选取最相似的前 top_k 个节点
                for row_idx in range(sim_batch.shape[0]):
                    row_sim = sim_batch[row_idx]
                    top_k_indices = np.argsort(row_sim)[-top_k:][::-1]
                    global_row = row_idx + i
                    global_cols = top_k_indices + j

                    current_indices = np.concatenate([
                        np.reshape([global_row] * top_k, [-1, 1]),
                        np.reshape(global_cols, [-1, 1])
                    ], axis=1)
                    indices.append(current_indices)

        return np.vstack(indices)

    similarity_matrix_tm = batch_cosine_similarity(transcriptome_features_pca, metabolite_features_pca,
                                                     batch_size=500,
                                                     top_k=3)

    similarity_matrix_mm = batch_cosine_similarity(metabolite_features, metabolite_features,
                                                     batch_size=5,
                                                     top_k=2)

    similarity_matrix_tt = batch_cosine_similarity(transcriptome_features_pca, transcriptome_features_pca,
                                                     batch_size=500,
                                                     top_k=3)

    similarity_matrix_tm[:, 0] = similarity_matrix_tm[:, 0] + gene_col_num
    similarity_matrix_tt = similarity_matrix_tt + gene_col_num

    edges_all = np.array(np.concatenate([gen_gen_edges_map, similarity_matrix_tm, similarity_matrix_mm, similarity_matrix_tt], axis=0))

    feature_all = np.concatenate([metabolite_features_pca, transcriptome_features_pca], axis=0)

    edge_index = edges_all.T

    values = np.ones(len(edges_all))

    adj_sp = torch.sparse_coo_tensor(indices=edge_index, values=torch.FloatTensor(values), size=[len(feature_all), len(feature_all)])

    all_indices = list(range(edge_index.shape[1]))

    random.shuffle(all_indices)

    # edge_index_1 = edges_all[all_indices[:int(0.5 * len(all_indices))]].T
    #
    # edge_index_2 = edges_all[all_indices[int(0.5 * len(all_indices)):]].T

    edge_index = torch.LongTensor(edge_index)
    # edge_index_1 = torch.LongTensor(edge_index_1)
    # edge_index_2 = torch.LongTensor(edge_index_2)
    feature_all = torch.FloatTensor(np.array(feature_all))
    #relations = torch.tensor(relations, dtype=torch.long)

    metab = gene_col_index
    trans = gene_row_index

    org_index = [metab, trans]

    return edge_index, feature_all, adj_sp, org_index, len(metab)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)



#
# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
