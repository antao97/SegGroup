''' SegGroup model

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

'''

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from chainer import cuda
from plyfile import PlyData

from data import ScanNet
from util import cross_entropy_loss, square_loss
from dataset.scannet.util import visualize_labels
import time

SEM_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
INS_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature1(x, k, idx=None, dim6=True):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)

    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim6:
            idx = knn(x[:, :3], k=k)
        else:
            idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0, 3, 1, 2)
    feature[:,:3] = feature[:,:3] - torch.mean(feature[:,:3], dim=-1, keepdim=True).repeat(1, 1, 1, k)
    feature[:,:3] *= 10
  
    return feature      # (batch_size, num_dims, num_points, k)


class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = get_graph_feature1(x, k=10)                 # (batch_size, 6, num_points) -> (batch_size, 6, num_points, k)
        x = self.conv1(x)                               # (batch_size, 6, num_points, k) -> (batch_size, 64, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]             # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]            # (batch_size, 64, num_points) -> (batch_size, 64)
        x2 = x.mean(dim=-1, keepdim=False)              # (batch_size, 64, num_points) -> (batch_size, 64)
        x = torch.cat([x1, x2], dim=-1)                 # (batch_size, 128)
        return x


def get_graph_feature2(x, idx):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    k = idx.size(-1)

    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
  
    return feature      # (batch_size, num_dims, num_points, k)


class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(9*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx):
        x = get_graph_feature2(x, idx)                  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                               # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]             # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        return x


class MLP3(nn.Module):
    def __init__(self):
        super(MLP3, self).__init__()
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(9*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx):
        x = get_graph_feature2(x, idx)                  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                               # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                               # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]             # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        return x


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, X, Edge):
        Edge_sum = Edge.sum(1, keepdim=True)
        Edge_norm = Edge / Edge_sum.repeat(1, Edge.shape[1])

        X = F.relu(self.fc(Edge_norm.mm(X)))
        return X


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(dim_in, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128, dim_out)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2) # (batch_size, dim_in) -> (batch_size, 128)
        x = self.dp1(x)
        x = self.linear2(x)                                             # (batch_size, 128) -> (batch_size, dim_out)
        return x


class DisjointSet():
    def __init__(self, weak_ins_label, weak_sem_label):
        self.size = weak_ins_label.shape[0]
        self.cluster_id = np.arange(self.size)
        self.indexs = [[i] for i in range(self.size)]
        self.weak_ins_label = np.array(weak_ins_label)
        self.weak_sem_label = np.array(weak_sem_label)
        self.point_num = np.ones(self.size)

    def find(self, idx):
        return self.cluster_id[idx]

    def union(self, id1, id2):
        if id1 == id2:
            return
        if (self.weak_ins_label[id1] != -1) and (self.weak_ins_label[id2] != -1) and (self.weak_ins_label[id1] != self.weak_ins_label[id2]):
            return
        self.cluster_id[self.indexs[id1]] = id2
        self.point_num[id2] += self.point_num[id1]
        if self.weak_ins_label[id1] != self.weak_ins_label[id2]:
            self.weak_ins_label[id2] = -self.weak_ins_label[id1]*self.weak_ins_label[id2]
            self.weak_sem_label[id2] = -self.weak_sem_label[id1]*self.weak_sem_label[id2]
        self.indexs[id2].extend(self.indexs[id1])
        self.indexs[id1] = []

    def connected(self, idx1, idx2):
        return self.find(idx1)==self.find(idx2)

    def get_point_num(self, idx):
        return self.point_num[self.find(idx)]

    def get_weak_ins_label(self, idx):
        return self.weak_ins_label[self.find(idx)]

    def get_weak_sem_label(self, idx):
        return self.weak_sem_label[self.find(idx)]

    def get_cluster_id(self, idx):
        return self.cluster_id[self.find(idx)]

    def get_cluster_list(self):
        cluster_list = []
        for i in range(self.size):
            if self.indexs[i] != []:
                cluster_list.append(self.indexs[i])
        return cluster_list


# Group nearby adjacency verts into new vert
def group_nearby_clusters(ds, Dist, adj, group_unmap, th):
    for i in range(len(adj)):
        if Dist[i] > th: 
            continue
        idx1 = group_unmap[adj[i][0].item()]
        idx2 = group_unmap[adj[i][1].item()]
        cluster_id1 = ds.find(idx1)
        cluster_id2 = ds.find(idx2)
        ds.union(cluster_id1, cluster_id2)

    while(1):
        has_small_vert = False
        for i in range(len(adj)):
            idx1 = group_unmap[adj[i][0].item()]
            idx2 = group_unmap[adj[i][1].item()]
            cluster_id1 = ds.find(idx1)
            cluster_id2 = ds.find(idx2)
            if (ds.point_num[cluster_id1] < 5) or (ds.point_num[cluster_id2] < 5):
                ds.union(cluster_id1, cluster_id2)
                has_small_vert = True
        if has_small_vert == False:
            break

    adj_connected = []
    adj_unconnected = []
    for i in range(len(adj)):
        idx1 = group_unmap[adj[i][0].item()]
        idx2 = group_unmap[adj[i][1].item()]
        if ds.connected(idx1, idx2):
            adj_connected.append(adj[i].unsqueeze(0))
        else:
            adj_unconnected.append(adj[i].unsqueeze(0))
    if len(adj_connected) > 0:
        adj_connected = torch.cat(adj_connected, dim=0)
    else:
        adj_connected = torch.LongTensor([])
    if len(adj_unconnected) > 0:
        adj_unconnected = torch.cat(adj_unconnected, dim=0)
    else:
        adj_unconnected = torch.LongTensor([])
    return ds, adj_connected, adj_unconnected


# Calculate similarity between adjacent verts
def calculate_similarity(Feat, adj, alpha=1):
    dists = calculate_distance(Feat, adj)
    sims = torch.exp(-dists*alpha)
    return sims


# Calculate distance between adjacent verts
def calculate_distance(Feat, adj):
    adj = adj.to(Feat.device)
    Feat1 = Feat[adj[:,0], :]
    Feat2 = Feat[adj[:,1], :]
    dists = F.pairwise_distance(Feat1, Feat2)
    return dists


# Get new cluster feature by max pooling and avg pooling
def aggregate_cluster_feature(Feat_old, clusters_new, use_avg=False):
    Feat_new = []
    for i, indexs in clusters_new.items():
        f1 = torch.max(Feat_old[indexs], dim=0, keepdim=True)[0]
        if use_avg:
            f2 = torch.mean(Feat_old[indexs], dim=0, keepdim=True)[0].unsqueeze(0)
            Feat_new.append(torch.cat([f1, f2], dim=-1))
        else:
            Feat_new.append(f1)
    Feat_new = torch.cat(Feat_new, dim=0)
    return Feat_new


def update_adj(adj_old, ds, cluster_unmap_old, cluster_map_new):
    adj_new = []
    for idx in adj_old:
        idx1 = cluster_map_new[ds.find(cluster_unmap_old[idx[0].item()])]
        idx2 = cluster_map_new[ds.find(cluster_unmap_old[idx[1].item()])]
        if idx1 == idx2:
            continue
        adj_new.append([idx1, idx2])
    adj_new = torch.LongTensor(adj_new)
    adj_new = torch.sort(adj_new, dim=-1)[0]
    adj_new = torch.unique(adj_new, dim=0)
    return adj_new


def build_similarity_matrix(sims, adj, size):
    sim_matrix = torch.eye(size).to(sims.device)
    sim_matrix[adj[:,0], adj[:,1]] = sims
    sim_matrix[adj[:,1], adj[:,0]] = sims
    return sim_matrix


def build_distance_matrix(dists, adj, size):
    dist_matrix = torch.ones(size, size).to(dists.device) * 1000
    dist_matrix[adj[:,0], adj[:,1]] = dists
    dist_matrix[adj[:,1], adj[:,0]] = dists
    return dist_matrix


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = xp.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = xp.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = xp.random.randint(pts.shape[1])
    else:
        indices[:, 0] = initial_idx

    batch_indices = xp.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = xp.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = xp.minimum(min_distances, dist)
    return indices, distances


def get_cluster_pointcloud(data, ds, point_num=128, transfrom=True):
    ds_list = ds.get_cluster_list()
    cluster_data_all = []
    for i in range(len(ds_list)):
        indexs = torch.LongTensor(ds_list[i])
        rep_num = point_num // indexs.shape[0]
        remainder_num = point_num % indexs.shape[0]
        if remainder_num > 0:
            choice, _ = farthest_point_sampling(np.array(data[indexs, :3].cpu()), remainder_num, initial_idx=0, skip_initial=True)
            if choice[0,-1] == 0:
                for j in range(1, choice[0].shape[0]+1):
                    if choice[0, -j] != 0:
                        break
                invalid_num = j - 1
                choice[0, -invalid_num:] = choice[0, :invalid_num]
            index_remainder = indexs[choice[0]]
            remainder = data[index_remainder, :]
        else:
            remainder = data[[], :]
        if rep_num != 0:
            cluster_data = torch.cat([data[indexs].repeat(rep_num, 1), remainder], dim=0)
        else:
            cluster_data = remainder
        if transfrom:
            cluster_data[:,:3] -= cluster_data[:,:3].mean(0)
            cluster_data[:,:3] /= torch.abs(cluster_data[:,:3]).max()
        cluster_data_all.append(cluster_data.unsqueeze(0))
    cluster_data_all = torch.cat(cluster_data_all, dim=0)
    return cluster_data_all


def combine_centralized_pointcloud(data, ds):
    ds_list = ds.get_cluster_list()
    data_center = data[:,:3].clone()
    for i in range(len(ds_list)):
        indexs = torch.LongTensor(ds_list[i])
        data_center[indexs] -= data_center[indexs].mean(0)
    data_new = torch.cat([data, data_center], dim=1)
    return data_new


def group_unlabeled_clusters(ds, Feat, adj, data):
    cluster_num_old = Feat.shape[0]
    ds_list_old = ds.get_cluster_list()
    cluster_unmap_old = {}
    for i in range(len(ds_list_old)):
        indexs = ds_list_old[i]
        cluster_unmap_old[i] = ds.find(indexs[0])

    while(1):
        dists = calculate_distance(Feat, adj)
        dist_matrix = build_distance_matrix(dists, adj, size=Feat.shape[0])

        index_min = torch.min(dist_matrix, dim=-1)[1].cpu()
        has_unlabeled_cluster = False
        for idx1 in range(dist_matrix.shape[0]):
            cluster1 = ds.find(cluster_unmap_old[idx1])
            if ds.weak_ins_label[cluster1] != -1:
                continue
            cluster2 = ds.find(cluster_unmap_old[index_min[idx1].item()])
            ds.union(cluster1, cluster2)

        ds_list_new = ds.get_cluster_list()
        cluster_map_new, cluster_unmap_new, cluster_new_to_old = {}, {}, {}
        for i in range(len(ds_list_new)):
            indexs = ds_list_new[i]
            cluster_map_new[ds.find(indexs[0])] = i
            cluster_unmap_new[i] = ds.find(indexs[0])
            cluster_new_to_old[i] = []
        for j in range(len(ds_list_old)):
            cluster_new_to_old[cluster_map_new[ds.find(cluster_unmap_old[j])]].append(j)
        adj = update_adj(adj, ds, cluster_unmap_old, cluster_map_new)
        Feat = aggregate_cluster_feature(Feat, cluster_new_to_old)
        ds_list_old = ds_list_new
        cluster_map_old = cluster_map_new
        cluster_unmap_old = cluster_unmap_new
        if Feat.shape[0] == cluster_num_old:
            break   
        else:
            cluster_num_old = Feat.shape[0]

    cluster_data = get_cluster_pointcloud(data, ds, point_num=1024, transfrom=False)[:,:,:3]
    has_unlabeled_cluster = False
    for i in range(Feat.shape[0]):
        cluster_id1 = ds.find(cluster_unmap_new[i])
        if ds.get_weak_ins_label(cluster_id1) != -1:
            continue
        has_unlabeled_cluster = True
        cluster_mean = torch.mean(cluster_data[i], dim=0).unsqueeze(0)
        dist_idx = torch.sort(torch.min(l2_norm(cluster_mean, cluster_data), dim=-1)[0])[1]
        for j in dist_idx.tolist():
            if i == j:
                continue
            cluster_id2 = ds.find(cluster_unmap_new[j])
            if ds.get_weak_ins_label(cluster_id2) == -1:
                continue
            ds.union(cluster_id1, cluster_id2)

    if has_unlabeled_cluster:
        ds_list_new = ds.get_cluster_list()
        cluster_map_new, cluster_unmap_new, cluster_new_to_old = {}, {}, {}
        for i in range(len(ds_list_new)):
            indexs = ds_list_new[i]
            cluster_map_new[ds.find(indexs[0])] = i
            cluster_unmap_new[i] = ds.find(indexs[0])
            cluster_new_to_old[i] = []
        for j in range(len(ds_list_old)):
            cluster_new_to_old[cluster_map_new[ds.find(cluster_unmap_old[j])]].append(j)
        adj = update_adj(adj, ds, cluster_unmap_old, cluster_map_new)
        Feat = aggregate_cluster_feature(Feat, cluster_new_to_old)

    return ds, Feat, adj


def get_knn(data, cluster, k=20):
    knn_all = torch.zeros(data.shape[0], k, dtype=torch.long)
    for i in range(len(cluster)):
        indexs = torch.LongTensor(cluster[i])
        if k >= indexs.shape[0]:
            idx = torch.arange(indexs.shape[0]).unsqueeze(0).repeat(indexs.shape[0], 1)
            knn_all[indexs, :indexs.shape[0]] = indexs[idx.view(-1)].view(-1, indexs.shape[0])
        else:
            idx = knn(data[indexs].unsqueeze(0).transpose(2, 1), k).squeeze(0)
            knn_all[indexs, :k] = indexs[idx.view(-1)].view(-1, k)
    return knn_all


def export_segment_label(ds, ds_unmap, output_root, unmap_path, layer, point_num=150000):
    ds_list = ds.get_cluster_list()
    seg_pred = torch.ones(point_num, dtype=torch.long) * -1
    for i in range(len(ds_list)):
        indexs = ds_list[i]
        label = ds.get_cluster_id(ds_unmap[i])
        seg_pred[indexs] = label.item()

    unmapping = torch.load(unmap_path)
    seg_pred_real = seg_pred[unmapping]

    seg_pred_output = []
    for i in range(seg_pred_real.shape[0]):
        seg_pred_output.append('%d\n' % seg_pred_real[i])

    if layer == 'final':
        output_path = os.path.join(output_root, 'final.seg.txt')
    else:
        output_path = os.path.join(output_root, 'layer_' + str(int(layer)) + '.seg.txt')

    with open(output_path, 'w') as f:
        f.writelines(seg_pred_output)
    f.close()

    return seg_pred_real


def export_instance_label(ds, ds_unmap, output_root, unmap_path, layer, point_num=150000):
    ds_list = ds.get_cluster_list()
    ins_pred = torch.ones(point_num, dtype=torch.long) * -1
    for i in range(len(ds_list)):
        indexs = ds_list[i]
        label = ds.get_weak_ins_label(ds_unmap[i])
        if label != -1:
            ins_pred[indexs] = label.item() + 1

    unmapping = torch.load(unmap_path)
    ins_pred_real = ins_pred[unmapping]

    ins_pred_output = []
    for i in range(ins_pred_real.shape[0]):
        ins_pred_output.append('%d\n' % ins_pred_real[i])

    if layer == 'final':
        output_path = os.path.join(output_root, 'final.ins.txt')
    else:
        output_path = os.path.join(output_root, 'layer_' + str(int(layer)) + '.ins.txt')

    with open(output_path, 'w') as f:
        f.writelines(ins_pred_output)
    f.close()

    return ins_pred_real


def export_semantic_label(ds, ds_unmap, output_root, unmap_path, layer, point_num=150000):
    ds_list = ds.get_cluster_list()
    sem_pred = torch.ones(point_num, dtype=torch.long) * -1
    for i in range(len(ds_list)):
        indexs = ds_list[i]
        label = ds.get_weak_sem_label(ds_unmap[i])
        if label != -1:
            sem_pred[indexs] = label.item() + 1

    unmapping = torch.load(unmap_path)
    sem_pred_real = sem_pred[unmapping]

    sem_pred_output = []
    for i in range(sem_pred_real.shape[0]):
        sem_pred_output.append('%d\n' % sem_pred_real[i])

    if layer == 'final':
        output_path = os.path.join(output_root, 'final.sem.txt')
    else:
        output_path = os.path.join(output_root, 'layer_' + str(int(layer)) + '.sem.txt')

    with open(output_path, 'w') as f:
        f.writelines(sem_pred_output)
    f.close()

    return sem_pred_real


def evaluate(scene_name, sem_pred, ins_pred):
    # Prepare for evaluation
    real_label_root = os.path.join('dataset', 'scannet', 'label', 'real', 'raw', scene_name)
    real_label_path = os.path.join(real_label_root, scene_name+'.label.pth')
    real_label = torch.load(real_label_path) 
    sem_true = real_label[:,0]
    ins_true = real_label[:,1]
    valid_idxs = torch.where(sem_true != 0)[0]
    sem_true = np.array(sem_true[valid_idxs])
    ins_true = np.array(ins_true[valid_idxs])
    sem_pred = np.array(sem_pred[valid_idxs])
    ins_pred = np.array(ins_pred[valid_idxs])

    # Calculate semantic mIoU
    I_sem = torch.zeros(1, 40)
    U_sem = torch.zeros(1, 40)
    for idx in range(40):
        sem = idx + 1
        I_sem[0, idx] += np.sum(np.logical_and(sem_pred == sem, sem_true == sem))
        U_sem[0, idx] += np.sum(np.logical_or(sem_pred == sem, sem_true == sem))
    IoU_sem = torch.cat([I_sem, U_sem], dim=0).unsqueeze(0)     # [1, 2, 40]

    # Calculate instance mIoU
    I_ins = torch.zeros(1, 40)
    U_ins = torch.zeros(1, 40)
    for ins in np.unique(ins_pred):
        if ins == -1:
            continue
        sem = sem_pred[np.where(ins_pred == ins)[0][0]]
        idx = sem - 1
        I_ins[0, idx] += np.sum(np.logical_and(ins_pred == ins, ins_true == ins))
        U_ins[0, idx] += np.sum(np.logical_or(ins_pred == ins, ins_true == ins))
    IoU_ins = torch.cat([I_ins, U_ins], dim=0).unsqueeze(0)     # [1, 2, 40]

    acc_sem = metrics.accuracy_score(sem_true, sem_pred)
    acc_ins = metrics.accuracy_score(ins_true, ins_pred)
    sem_valid_idxs = []
    for sem in SEM_VALID_CLASS_IDS:
        sem_valid_idxs.append(np.where(sem_true == sem)[0])
    sem_valid_idxs = np.concatenate(sem_valid_idxs, axis=0)
    acc_sem_selected = metrics.accuracy_score(sem_true[sem_valid_idxs], sem_pred[sem_valid_idxs])
    ins_valid_idxs = []
    for ins in INS_VALID_CLASS_IDS:
        ins_valid_idxs.append(np.where(ins_true == ins)[0])
    ins_valid_idxs = np.concatenate(ins_valid_idxs, axis=0)
    acc_ins_selected = metrics.accuracy_score(ins_true[ins_valid_idxs], ins_pred[ins_valid_idxs])
    acc = torch.Tensor([acc_sem, acc_ins, acc_sem_selected, acc_ins_selected])  # 4
    return IoU_sem, IoU_ins, acc


class SegModel(nn.Module):
    def __init__(self, exp_name='exp', cuda=True, visualize=False, sem_infer=False, ins_infer=False):
        super(SegModel, self).__init__()
        self.exp_name = exp_name
        self.cuda = cuda
        if self.cuda:
            self.device = torch.device('cuda')
        self.visualize = visualize
        self.sem_infer = sem_infer
        self.ins_infer = ins_infer

        self.data_root = os.path.join('dataset', 'scannet')
        scene_list_path = os.path.join(self.data_root, 'scannetv2_train.txt')
        with open(scene_list_path, 'r') as f:
            self.scene_list = f.readlines()
        f.close()
        
        self.epoch = '0'
        self.mlp_1 = MLP1() # out_dim 64
        self.mlp_2 = MLP2() # out_dim 64
        self.gcn_2 = GCN(dim_in=192, dim_out=192)
        self.mlp_3 = MLP3() # out_dim 64
        self.gcn_3 = GCN(dim_in=256, dim_out=256)
        self.classifier = Classifier(dim_in=256, dim_out=40)


    def forward(self, data, weak_label, info):
        data, weak_label, info = data[0], weak_label[0], info[0]
        self.point_num = data.shape[0]
        scene_name = self.scene_list[info][:-1]
        if self.epoch in ['sem_infer', 'ins_infer']:
            output_root = os.path.join('results', self.exp_name, scene_name, self.epoch)
        else:
            output_root = os.path.join('results', self.exp_name, scene_name, 'epoch_'+self.epoch)
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # Initialize data paths
        adj_path = os.path.join('dataset/scannet/adj/mesh/resampled', scene_name, scene_name+'.adj.pth')
        map_path = os.path.join('dataset/scannet/data/resampled', scene_name, scene_name+'.map.pth')
        unmap_path = os.path.join('dataset/scannet/data/resampled', scene_name, scene_name+'.unmap.pth')
        ds_path = os.path.join('dataset/scannet/label/real/resampled', scene_name, scene_name+'.seg.json')
        mesh_path = os.path.join('/data1/antao/Documents/Datasets/ScanNet_raw', \
                                        'scans', scene_name, scene_name+'_vh_clean_2.ply')

        # Load mesh file for visualization
        if self.visualize:
            with open(mesh_path, 'rb') as f:
                plydata = PlyData.read(f)
            f.close()


        ### Graph Initialization
        # Initialize disjoint set
        ds = DisjointSet(weak_label[:,1].cpu(), weak_label[:,0].cpu())
        with open(ds_path, 'r') as f:
            ds.indexs = json.load(f)
        f.close()
        for idx_list in ds.indexs:
            if idx_list == []:
                continue
            id = idx_list[0]
            ds.cluster_id[idx_list] = id
            ds.point_num[id] = len(idx_list)
        
        # Initialize adjacency graph
        adj_0 = torch.load(adj_path)
        cluster_unmap_0 = {i: i for i in range(data.shape[0])}
        ds_list_1 = ds.get_cluster_list()
        cluster_1, cluster_map_1, cluster_unmap_1 = {}, {}, {}
        for i in range(len(ds_list_1)):
            indexs = ds_list_1[i]
            cluster_1[i] = indexs
            cluster_map_1[ds.find(indexs[0])] = i
            cluster_unmap_1[i] = ds.find(indexs[0])
        adj_1 = update_adj(adj_0, ds, cluster_unmap_0, cluster_map_1)

        # Export results
        export_segment_label(ds, cluster_unmap_1, output_root, unmap_path, layer=1)
        export_instance_label(ds, cluster_unmap_1, output_root, unmap_path, layer=1)
        export_semantic_label(ds, cluster_unmap_1, output_root, unmap_path, layer=1)
        if self.visualize:
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_1.seg.txt'), label_type='segment', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_1.ins.txt'), label_type='instance', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_1.sem.txt'), label_type='semantic', plydata=plydata)


        ### Structural Grouping Layer
        # MLP
        data_1 = get_cluster_pointcloud(data, ds, point_num=64)
        Feat_mlp_1 = self.mlp_1(data_1.transpose(2,1))
        Feat_1 = Feat_mlp_1

        # Clustering
        dists_1 = calculate_distance(Feat_1, adj_1)
        # print(np.unique(ds.cluster_id).shape[0])
        if self.sem_infer:
            ds, adj_connected_1, adj_unconnected_1 = group_nearby_clusters(ds, dists_1, adj_1, cluster_unmap_1, th=3)
        else:
            ds, adj_connected_1, adj_unconnected_1 = group_nearby_clusters(ds, dists_1, adj_1, cluster_unmap_1, th=6)
        # print(np.unique(ds.cluster_id).shape[0])
        ds_list_2 = ds.get_cluster_list()
        cluster_2, cluster_map_2, cluster_unmap_2, cluster_2_to_1 = {}, {}, {}, {}
        for i in range(len(ds_list_2)):
            indexs = ds_list_2[i]
            cluster_2[i] = indexs
            cluster_map_2[ds.find(indexs[0])] = i
            cluster_unmap_2[i] = ds.find(indexs[0])
            cluster_2_to_1[i] = []
        for j in range(len(ds_list_1)):
            cluster_2_to_1[cluster_map_2[ds.find(cluster_unmap_1[j])]].append(j)
        adj_2 = update_adj(adj_unconnected_1, ds, cluster_unmap_1, cluster_map_2)
        Feat_2 = aggregate_cluster_feature(Feat_1, cluster_2_to_1)

        # Export results
        export_segment_label(ds, cluster_unmap_2, output_root, unmap_path, layer=2)
        ins_pred = export_instance_label(ds, cluster_unmap_2, output_root, unmap_path, layer=2)
        sem_pred = export_semantic_label(ds, cluster_unmap_2, output_root, unmap_path, layer=2)
        if self.visualize:
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_2.seg.txt'), label_type='segment', shuffle=True, plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_2.ins.txt'), label_type='instance', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_2.sem.txt'), label_type='semantic', plydata=plydata)

        if self.sem_infer:
            IoU_sem, IoU_ins, acc = evaluate(scene_name, sem_pred, ins_pred)
            return IoU_sem.to(data.device), IoU_ins.to(data.device), acc.to(data.device)


        ### Semantic Grouping Layer 1
        # MLP
        knn_2 = get_knn(data[:,:3].cpu(), cluster_2, k=20)
        knn_2 = knn_2.to(data.device)
        data_2 = combine_centralized_pointcloud(data, ds)
        Feat_mlp_2 = self.mlp_2(data_2.transpose(1,0).unsqueeze(0), knn_2.unsqueeze(0))
        Feat_mlp_2 = Feat_mlp_2.squeeze(0).transpose(1,0).contiguous()
        Feat_mlp_2 = aggregate_cluster_feature(Feat_mlp_2, cluster_2)
        Feat_2 = torch.cat([Feat_2, Feat_mlp_2], dim=-1)

        # GCN
        sims_2 = calculate_similarity(Feat_2, adj_2, alpha=1/8)
        sim_matrix_2 = build_similarity_matrix(sims_2, adj_2, size=Feat_2.shape[0])
        Feat_2 = self.gcn_2(Feat_2, sim_matrix_2)

        # Clustering
        dists_2 = calculate_distance(Feat_2, adj_2)
        ds, adj_connected_2, adj_unconnected_2 = group_nearby_clusters(ds, dists_2, adj_2, cluster_unmap_2, th=2)
        ds_list_3 = ds.get_cluster_list()
        cluster_3, cluster_map_3, cluster_unmap_3, cluster_3_to_2 = {}, {}, {}, {}
        for i in range(len(ds_list_3)):
            indexs = ds_list_3[i]
            cluster_3[i] = indexs
            cluster_map_3[ds.find(indexs[0])] = i
            cluster_unmap_3[i] = ds.find(indexs[0])
            cluster_3_to_2[i] = []
        for j in range(len(ds_list_2)):
            cluster_3_to_2[cluster_map_3[ds.find(cluster_unmap_2[j])]].append(j)
        adj_3 = update_adj(adj_unconnected_2, ds, cluster_unmap_2, cluster_map_3)
        Feat_3 = aggregate_cluster_feature(Feat_2, cluster_3_to_2)

        # Export results
        export_segment_label(ds, cluster_unmap_3, output_root, unmap_path, layer=3)
        export_instance_label(ds, cluster_unmap_3, output_root, unmap_path, layer=3)
        export_semantic_label(ds, cluster_unmap_3, output_root, unmap_path, layer=3)
        if self.visualize:
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_3.seg.txt'), label_type='segment', shuffle=True, plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_3.ins.txt'), label_type='instance', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_3.sem.txt'), label_type='semantic', plydata=plydata)


        ### Semantic Grouping Layer 2
        # MLP
        knn_3 = get_knn(data[:,:3].cpu(), cluster_3, k=20)
        knn_3 = knn_3.to(data.device)
        data_3 = combine_centralized_pointcloud(data, ds)
        Feat_mlp_3 = self.mlp_3(data_3.transpose(1,0).unsqueeze(0), knn_3.unsqueeze(0))
        Feat_mlp_3 = Feat_mlp_3.squeeze(0).transpose(1,0).contiguous()
        Feat_mlp_3 = aggregate_cluster_feature(Feat_mlp_3, cluster_3)
        Feat_3 = torch.cat([Feat_3, Feat_mlp_3], dim=-1)

        # GCN
        sims_3 = calculate_similarity(Feat_3, adj_3, alpha=1/8)
        sim_matrix_3 = build_similarity_matrix(sims_3, adj_3, size=Feat_3.shape[0])
        Feat_3 = self.gcn_3(Feat_3, sim_matrix_3)

        # Clustering
        dists_3 = calculate_distance(Feat_3, adj_3)
        ds, adj_connected_3, adj_unconnected_3 = group_nearby_clusters(ds, dists_3, adj_3, cluster_unmap_3, th=2)
        ds_list_4 = ds.get_cluster_list()
        cluster_4, cluster_map_4, cluster_unmap_4, cluster_4_to_3 = {}, {}, {}, {}
        for i in range(len(ds_list_4)):
            indexs = ds_list_4[i]
            cluster_4[i] = indexs
            cluster_map_4[ds.find(indexs[0])] = i
            cluster_unmap_4[i] = ds.find(indexs[0])
            cluster_4_to_3[i] = []
        for j in range(len(ds_list_3)):
            cluster_4_to_3[cluster_map_4[ds.find(cluster_unmap_3[j])]].append(j)
        adj_4 = update_adj(adj_unconnected_3, ds, cluster_unmap_3, cluster_map_4)
        Feat_4 = aggregate_cluster_feature(Feat_3, cluster_4_to_3)

        # Export results
        export_segment_label(ds, cluster_unmap_4, output_root, unmap_path, layer=4)
        export_instance_label(ds, cluster_unmap_4, output_root, unmap_path, layer=4)
        export_semantic_label(ds, cluster_unmap_4, output_root, unmap_path, layer=4)
        if self.visualize:
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_4.seg.txt'), label_type='segment', shuffle=True, plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_4.ins.txt'), label_type='instance', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'layer_4.sem.txt'), label_type='semantic', plydata=plydata)


        ### Final Clusering
        # Grouping all
        ds, Feat_5, adj_5 = group_unlabeled_clusters(ds, Feat_4, adj_4, data.cpu())
        ds_list_5 = ds.get_cluster_list()
        cluster_5, cluster_map_5, cluster_unmap_5, cluster_5_to_4 = {}, {}, {}, {}
        # list = np.arange(len(ds_list_5))
        # np.random.seed(1)
        # np.random.shuffle(list)
        for i in range(len(ds_list_5)):
        # for i in list:
            indexs = ds_list_5[i]
            cluster_5[i] = indexs
            cluster_map_5[ds.find(indexs[0])] = i
            cluster_unmap_5[i] = ds.find(indexs[0])
            cluster_5_to_4[i] = []
        for j in range(len(ds_list_4)):
            cluster_5_to_4[cluster_map_5[ds.find(cluster_unmap_4[j])]].append(j)

        # Export results
        ins_pred = export_instance_label(ds, cluster_unmap_5, output_root, unmap_path, layer='final')
        sem_pred = export_semantic_label(ds, cluster_unmap_5, output_root, unmap_path, layer='final')
        if self.visualize:
            visualize_labels(mesh_path, os.path.join(output_root, 'final.ins.txt'), label_type='instance', plydata=plydata)
            visualize_labels(mesh_path, os.path.join(output_root, 'final.sem.txt'), label_type='semantic', plydata=plydata)  

        # Evaluate Pseudo Labels
        IoU_sem, IoU_ins, acc = evaluate(scene_name, sem_pred, ins_pred)

        if self.ins_infer:
            return IoU_sem.to(data.device), IoU_ins.to(data.device), acc.to(data.device)


        ### Classifier
        # Get per-instance features
        ins_list = []
        sem_list = []
        for i in range(Feat_5.shape[0]):
            ins = ds.get_weak_ins_label(cluster_unmap_5[i]).item()
            sem = ds.get_weak_sem_label(cluster_unmap_5[i]).item()
            ins_list.append(ins)
            sem_list.append(sem)
        ins_gt = np.unique(ins_list)
        sem_gt = []
        Feat_6 = []
        for ins in ins_gt:
            indexs = np.where(ins_list == ins)[0]
            sem = sem_list[indexs[0]]
            if indexs.shape[0] > 1:
                Feat_6.append(torch.max(Feat_5[indexs], dim=0, keepdim=True)[0])
            else:
                Feat_6.append(Feat_5[indexs])
            sem_gt.append(sem)
        Feat_6 = torch.cat(Feat_6, dim=0)
        sem_gt = torch.LongTensor(sem_gt).to(Feat_6.device)

        # Classify final features
        Feat_sem = self.classifier(Feat_6)

        # Calculate cross-entropy loss
        loss_sum = cross_entropy_loss(Feat_sem, sem_gt).unsqueeze(0)
        loss_num = torch.Tensor([Feat_sem.shape[0]]).to(loss_sum.device)    # 1
        loss = torch.cat([loss_sum, loss_num]).unsqueeze(0)     # [1, 2]

        
        return loss, IoU_sem.to(loss.device), IoU_ins.to(loss.device), acc.to(loss.device)
