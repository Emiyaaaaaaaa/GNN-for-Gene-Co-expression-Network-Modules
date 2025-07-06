import os
import glob
import random
import torch
import random
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch.nn.functional as F
from torch.autograd import Variable
from untily.argparess import create_parser
from models_our import GenCENet
from sklearn.metrics import silhouette_score

import time

from untily.utilsall import load_data


args = create_parser()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# loss_val_all = []
best_loss = args.epochs + 1
best_epoch = 0

if args.cuda:
    print('cuda is avaliable...')
    torch.cuda.manual_seed(args.seed)
else:
    print('cuda is not avaliable...')


select = 'GenCENet'  # GenCENet


edge_index, features, adj_sp, org_index, n1 = load_data()

print('model:{}'.format(select))
if select == 'GenCENet':
    model = GenCENet(
        nfeat=features.shape[1],
        nclass=64,
        nhid=args.hidden,
        adj=adj_sp.cuda()
    )
    keypt = 0

gpu_cuda = torch.cuda.is_available()
lamda = 0.9

edge_train = edge_index[:, range(int(edge_index.shape[1] * lamda))]
edge_train_label = torch.ones(edge_train.shape[1])

edge_test = edge_index[:, range(int(edge_index.shape[1] * lamda), int(edge_index.shape[1]))]
edge_test_label = torch.ones(edge_test.shape[1])

def generate_negative_samples(edge_index, num_nodes, sample_number, n1):
    # 将边索引转换为 numpy 数组并存储为正边列表
    pos_edges = np.reshape(edge_index.cpu().numpy(), [-1, 2])
    # 将正边转换为集合，方便快速查找
    pos_edge_set = set(tuple(edge) for edge in pos_edges)
    pos_edge_set.update(tuple(edge[::-1]) for edge in pos_edges)

    # 初始化负边列表
    negative_edges = []

    # 计算每个节点的邻居节点
    neighbors = [set() for _ in range(num_nodes)]
    for u, v in pos_edges:
        neighbors[u].add(v)
        neighbors[v].add(u)

    while len(negative_edges) < sample_number:
        percentage = len(negative_edges) / sample_number * 100
        print('='*20, f"{percentage:.2f}%", '='*20)
        # 随机选择一个节点
        node1 = np.random.randint(num_nodes)
        # 从非邻居节点中选择一个节点
        non_neighbors = list(set(range(num_nodes)) - neighbors[node1] - {node1})
        if non_neighbors:
            node2 = np.random.choice(non_neighbors)

            negative_edge = (node1, node2)
            if negative_edge not in pos_edge_set:
                negative_edges.append(negative_edge)


    negative_edges = np.array(negative_edges)
    print(negative_edges)
    return negative_edges


# edge_index 边索引张量，num_nodes 是图中节点的数量
num_nodes = features.shape[0]
train_sample = edge_train.shape[1]
train_sample = 10
negative_samples = torch.tensor(generate_negative_samples(edge_index, num_nodes, train_sample, n1).T)  ###

edge_train_label_neg = torch.zeros(negative_samples.shape[1])

test_sample = edge_test.shape[1]
test_sample = 10
negative_test = torch.tensor(generate_negative_samples(edge_index, num_nodes, test_sample, n1).T)  ###
edge_test_label_neg = torch.zeros(negative_test.shape[1])

edge_train_end = torch.cat([edge_train, negative_samples], dim=1)
edge_train_label_end = torch.cat([edge_train_label, edge_train_label_neg], dim=0)

edge_test_end = torch.cat([edge_test, negative_test], dim=1)
edge_test_label_end = torch.cat([edge_test_label, edge_test_label_neg], dim=0)

numbers_list_train = list(range(edge_train_end.shape[1]))
random.shuffle(numbers_list_train)

numbers_list_test = list(range(edge_test_end.shape[1]))
random.shuffle(numbers_list_test)

edge_train_end = edge_train_end[:, numbers_list_train]
edge_train_label_end = edge_train_label_end[numbers_list_train]
edge_test_end = edge_test_end[:, numbers_list_test]
edge_test_label_end = edge_test_label_end[numbers_list_test]

# gpu_cuda = False
if gpu_cuda:
    print('data to GPU')
    model = model.cuda()
    features = features.cuda()

    edge_index = edge_index.cuda()
    edge_train = edge_train.cuda()
    edge_test = edge_test.cuda()

    edge_train_end = edge_train_end.cuda()
    edge_train_label_end = edge_train_label_end.cuda()
    edge_test_end = edge_test_end.cuda()
    edge_test_label_end = edge_test_label_end.cuda()

adj = edge_train_end
adj_test = edge_test_end

adj, features = Variable(adj), Variable(features)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_function = torch.nn.CrossEntropyLoss()

train_loss = []
test_loss = []

epoch_list = []
out_loss_list = []
for epoch in range(args.epochs):
    time.time()
    model.train()
    optimizer.zero_grad()

    output, cl_loss = model(features, adj)

    pre_trains = torch.sigmoid((output[edge_train_end[0]] * output[edge_train_end[1]]).sum(dim=1))

    loss_train = F.binary_cross_entropy_with_logits(
        pre_trains,
        edge_train_label_end
    ) + cl_loss
    ############################################
    total_samples = edge_train_label_end.size(0)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output, cl_loss = model(features, adj)
    pre_auc = (torch.sigmoid((output[edge_test_end[0]] * output[edge_test_end[1]]).sum(dim=1))).float()

    loss_test = F.binary_cross_entropy_with_logits(
        pre_auc,
        edge_test_label_end
    ) + cl_loss

    print(
        'epoch={:04d}'.format(epoch + 1),
        'loss_train={:.4f}'.format(loss_train.data.item())
    )

    train_loss.append(loss_train.data.item())
    test_loss.append(loss_test.data.item())
    epoch_list.append(epoch)

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))

    if test_loss[-1] < best_loss:
        best_loss = test_loss[-1]
        best_epoch = epoch
    else:
        pass

    files = glob.glob('*.pkl')
    for file in files:
        epoch_ = int(file.split('.')[0])
        if epoch_ < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_ = int(file.split('.')[0])
    if epoch_ > best_epoch:
        os.remove(file)



import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(10, 7))

# 在第一个子图中绘制损失曲线
axs[0].plot(epoch_list, train_loss, label='Loss', color='orange')
axs[0].set_title('Train_loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Train_loss')
axs[0].legend()

# 在第二个子图中绘制准确率曲线
axs[1].plot(epoch_list[keypt:], test_loss[keypt:], label='Accuracy', color='lightblue')
axs[1].set_title('Test_loss')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Test_loss')
axs[1].legend()

# 调整子图之间的间距
plt.tight_layout()

# 显示图表
plt.show()

model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
# 测试
model.eval()
end_adj = torch.cat([edge_train, edge_test], dim=1)
output_best, _ = model(features, end_adj)

np.savetxt(r'output_best.txt', output_best.detach().cpu().numpy()[n1:], fmt="%f", delimiter=',')
output_best = output_best.detach().cpu().numpy()[n1:]
# 下面为自动识别模型的聚类的最优类别数   为自动寻找过程  较为缓慢

print('clusters optimizering...')

# 初始化评价指标列表
silhouette_scores = []
strat_num = 10  # 最小簇数
max_clusters = 20  # 最大簇数
cluster_range = range(strat_num, max_clusters + 1)  # 簇数范围：10到20

# 计算轮廓系数（注意：轮廓系数对簇数=1无定义，所以起始簇数≥2）
for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)  # 增加n_init避免局部最优
    labels = kmeans.fit_predict(output_best)
    silhouette_avg = silhouette_score(output_best, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"簇数={n_clusters}，轮廓系数={silhouette_avg:.4f}")

print('轮廓系数列表:', silhouette_scores)


# ==================== 优化最优簇数选择逻辑 ====================
def find_best_silhouette(sil_scores, cluster_range):
    max_score_idx = np.argmax(sil_scores)
    return cluster_range[max_score_idx]

# 选择最优簇数
optimal_k = find_best_silhouette(silhouette_scores, list(cluster_range))
print(f'基于轮廓系数的最优簇数: {optimal_k}')


# ==================== 绘制轮廓系数折线图 ====================
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of clusters', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('clusters', fontsize=14)
plt.xticks(cluster_range)  # 确保x轴刻度与簇数一致
plt.grid(alpha=0.3)  # 增加网格线，便于观察
plt.tight_layout()  # 自动调整布局
plt.show()


# ==================== 聚类与可视化 ====================
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
labels = kmeans.fit_predict(output_best)

# PCA降维用于可视化
pca = PCA(n_components=2)
node_vectors_pca = pca.fit_transform(output_best)

# 绘制聚类散点图
plt.figure(figsize=(10, 8))  # 增大图像尺寸，更清晰
scatter = plt.scatter(
    node_vectors_pca[:, 0],
    node_vectors_pca[:, 1],
    c=labels,
    cmap='viridis',  # 颜色映射，可替换为'rainbow'等
    s=50,  # 点大小
    alpha=0.8,  # 透明度，避免点重叠
    edgecolors='k',  # 点边缘颜色，增强区分度
    linewidths=0.5
)
plt.colorbar(scatter, label='clusters')  # 颜色条标注
plt.title(f'KMeans k={optimal_k}', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.savefig('cluster_result.png', dpi=300, bbox_inches='tight')

print('finished')