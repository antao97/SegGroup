''' Utility functions for prepare ScanNet dataset

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

'''

import os
import csv
import json
import random
import torch
import numpy as np
from plyfile import PlyData
from chainer import cuda

SEM_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
INS_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

num_colors = 40
colors = [
       (255, 255, 255),     # unlabeled 0
       (174, 199, 232),     # wall 1
       (152, 223, 138),     # floor 2
       (31, 119, 180),      # cabinet 3
       (255, 187, 120),     # bed 4
       (188, 189, 34),      # chair 5
       (140, 86, 75),       # sofa 6
       (255, 152, 150),     # table 7
       (214, 39, 40),       # door 8
       (197, 176, 213),     # window 9
       (148, 103, 189),     # bookshelf 10
       (196, 156, 148),     # picture 11
       (23, 190, 207),      # counter 12
       (178, 76, 76),  
       (247, 182, 210),     # desk 14
       (66, 188, 102), 
       (219, 219, 141),     # curtain 16
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),      # refrigerator 24
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),     # shower curtain 28
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),       # toilet 33
       (112, 128, 144),     # sink 34
       (96, 207, 209), 
       (227, 119, 194),     # bathtub 36
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),       # otherfurn 39
       (100, 85, 144),
    ]


mapper = np.ones(42) * 20
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    mapper[x] = i
mapper[0] = -1
mapper[-1] = -1


def load_labels(label_path):
    if label_path.endswith('.txt'):
        with open(label_path, 'r') as f:
            file = f.readlines()
        f.close()
        labels = []
        for i in range(len(file)):
            labels.append(int(file[i][:-1]))
    elif label_path.endswith('.json'):
        with open(label_path, 'r') as f:
            file = json.load(f)
        f.close()
        labels = file["segIndices"]
    else:
        print('Not supported file type!')
        exit(1)
    return labels


def load_seg_labels(label_file):
    with open(label_file, 'r') as f:
        file = json.load(f)
    f.close()
    labels = file["segIndices"]
    return labels


def read_label_mapper(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapper = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapper[row[label_from]] = int(row[label_to])
    return mapper


def load_aggregation(aggregation_file, mapper):
    with open(aggregation_file, 'r') as f:
        file = json.load(f)
    f.close()
    seg2ins = {}
    seg2sem = {}
    for seg in file["segGroups"]:
        if (aggregation_file.split('/')[-1][:12] == 'scene0217_00') and (seg['objectId'] == 31):
            break
        for seg_label in seg['segments']:
            seg2ins.update({seg_label: seg['objectId']+1})
            seg2sem.update({seg_label: mapper[seg['label']]})
    return seg2ins, seg2sem


# Generate real instance and semantic labels in .txt
def generate_real_labels(scene_path):
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]
    scene_name = os.path.split(scene_path)[-1]
    seg_path = os.path.join(scene_path, scene_name + '_vh_clean_2.0.010000.segs.json')
    aggregation_path = os.path.join(scene_path, scene_name + '.aggregation.json')
    mapper_path = os.path.join('/'.join(scene_path.split('/')[:-2]), 'scannetv2-labels.combined.tsv')
    seg_labels = load_seg_labels(seg_path)

    mapper = read_label_mapper(mapper_path)
    seg2ins, seg2sem = load_aggregation(aggregation_path, mapper)

    ins_labels = []
    sem_labels = []
    for seg_label in seg_labels:
        if seg_label in seg2ins.keys():
            ins_labels.append(str(seg2ins[seg_label])+'\n')
            sem_labels.append(str(seg2sem[seg_label])+'\n')
        else:
            ins_labels.append('0\n')
            sem_labels.append('0\n')

    '''
    semantic label:
        unlabeled:  0
        labeled:    1 ~ 40
    instance label: 
        unlabeled:  0
        labeled:    1 ~ instance num
    '''

    output_path = os.path.join('label', 'real', 'raw', scene_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ins_label_path = os.path.join(output_path, scene_name + '.ins.txt')
    sem_label_path = os.path.join(output_path, scene_name + '.sem.txt')
    with open(ins_label_path, 'w') as f:
        f.writelines(ins_labels)
    f.close()
    with open(sem_label_path, 'w') as f:
        f.writelines(sem_labels)
    f.close()


# Generate segment labels in .txt and disjoint-set in .json
def generate_seg_labels_and_ds_set(scene_path):
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]
    scene_name = os.path.split(scene_path)[-1]
    seg_path = os.path.join(scene_path, scene_name + '_vh_clean_2.0.010000.segs.json')
    
    seg_labels = load_seg_labels(seg_path)
    seg_labels_unique = np.unique(seg_labels)
    seg_remapper = np.zeros(seg_labels_unique.max()+1)
    for i, x in enumerate(seg_labels_unique):
        seg_remapper[x] = i

    seg_labels = seg_remapper[seg_labels]

    '''
    segment label: 
        0 ~ (segment num - 1)
    '''

    seg_output = []
    for i in range(seg_labels.shape[0]):
        seg_output.append(str(int(seg_labels[i]))+'\n')

    output_root = os.path.join('label', 'real', 'raw', scene_name)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    seg_label_path = os.path.join(output_root, scene_name + '.seg.txt')
    with open(seg_label_path, 'w') as f:
        f.writelines(seg_output)
    f.close()

    mapper = np.array(torch.load(os.path.join('data', 'resampled', scene_name, scene_name+'.map.pth')))
    seg_labels_sampled = seg_labels[mapper]

    # build disjoint-set for segments
    ds_clusters = [[] for i in range(seg_labels_sampled.shape[0])]
    for seg in np.unique(seg_labels_sampled):
        indexs = np.where(seg_labels_sampled == seg)[0].tolist()
        ds_clusters[indexs[0]] = indexs

    output_path = os.path.join('label', 'real', 'resampled', scene_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ds_path = os.path.join(output_path, scene_name+'.seg.json')
    with open(ds_path, 'w') as f:
        json.dump(ds_clusters, f)
    f.close()


# Generate segment adjacency matrix
def generate_seg_adjacency_matrix(mesh_path, seg_path, plydata=None):
    scene_name = mesh_path.split('/')[-2]
    seg_labels = load_labels(seg_path)
    if plydata is None:
        with open(mesh_path, 'rb') as f:
            plydata = PlyData.read(f)
        f.close()

    seg_num = max(seg_labels)+1
    adjacency_matrix = np.zeros([seg_num, seg_num])

    num_faces = plydata['face'].count
    for i in range(num_faces):
        face = plydata['face']['vertex_indices'][i]
        for idx in [[0,1], [0,2], [1,2]]:
            seg1 = seg_labels[face[idx[0]]]
            seg2 = seg_labels[face[idx[1]]]
            if seg1 != seg2:
                adjacency_matrix[seg1][seg2] = 1
                adjacency_matrix[seg2][seg1] = 1
    return adjacency_matrix


def find_index(clusters, id):
    for index in range(len(clusters)):
        if id in clusters[index]:
            return index

def group_adjacency_segs(adjacency_matrix, segs):
    clusters = [[id] for id in segs]
    for i in range(len(segs)):
        for j in range(i):
            id1 = segs[i]
            id2 = segs[j]
            if adjacency_matrix[id1,id2] == 0:
                continue
            index1 = find_index(clusters, id1)
            index2 = find_index(clusters, id2)
            if index1 != index2:
                clusters[index1].extend(clusters[index2])
                clusters.pop(index2)
    return clusters

# Generate weak labels in .txt
def generate_weak_labels(scene_path, plydata=None, label_style='manual', manual_label_path=None, main_num=-1, anno_num=1):   
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]
    scene_name = os.path.split(scene_path)[-1]
    mesh_path = os.path.join(scene_path, scene_name + '_vh_clean_2.ply')
    ins_path = os.path.join('label', 'real', 'raw', scene_name, scene_name + '.ins.txt')
    sem_path = os.path.join('label', 'real', 'raw', scene_name, scene_name + '.sem.txt')
    seg_path = os.path.join('label', 'real', 'raw', scene_name, scene_name + '.seg.txt')

    ins_labels = np.array(load_labels(ins_path))
    sem_labels = np.array(load_labels(sem_path))
    ins_unique = np.unique(ins_labels)
    seg_label_id_list = []

    if label_style == 'manual':
        seg_path = os.path.join(scene_path, scene_name + '_vh_clean_2.0.010000.segs.json')
        seg_labels = np.array(load_seg_labels(seg_path))
        seg_unique = np.unique(seg_labels)

        with open(os.path.join(manual_label_path, scene_name+'.json'), 'r') as f:
            manual_labels = json.load(f)
        f.close()

        for ins in manual_labels:
            for seg in manual_labels[ins]:
                seg_label_id_list.append(int(seg))

    else:
        seg_labels = np.array(load_labels(seg_path))
        seg_unique = np.unique(seg_labels)

        if plydata is None:
            with open(mesh_path, 'rb') as f:
                plydata = PlyData.read(f)
            f.close()

        adjacency_matrix = generate_seg_adjacency_matrix(mesh_path, seg_path, plydata)
        
        ins2seg = {}
        ins_num = ins_unique.shape[0]
        for i in range(ins_unique.shape[0]):
            ins = ins_unique[i]
            if ins == 0:
                continue    # ignore unlabeled segments
            ins_indexs = np.where(ins_labels==ins)[0]
            ins_segs = np.unique(seg_labels[ins_indexs])
            clusters = group_adjacency_segs(adjacency_matrix, ins_segs)
            cluster_point_num = []
            cluster_main_seg_id = []
            cluster_main_seg_point_num = []
            for j in range(len(clusters)):
                seg_point_num = []
                for seg_id in clusters[j]:
                    seg_point_num.append(np.where(seg_labels==seg_id)[0].shape[0])
                cluster_point_num.append(np.sum(seg_point_num))
                if main_num != -1:
                    seg_sort_id = np.array(clusters[j])[np.argsort(-np.array(seg_point_num))[:main_num]]
                    seg_sort_point_num = -np.sort(-np.array(seg_point_num))[:main_num]
                else:
                    seg_sort_id = np.array(clusters[j])[np.argsort(-np.array(seg_point_num))]
                    seg_sort_point_num = -np.sort(-np.array(seg_point_num))
                cluster_main_seg_id.append(seg_sort_id)
                cluster_main_seg_point_num.append(seg_sort_point_num)
            cluster_determin_index = np.argmax(cluster_point_num)
            if label_style == 'maxseg':
                for i in range(anno_num):
                    if i < len(cluster_main_seg_id[cluster_determin_index]):
                        seg_label_id_list.append(cluster_main_seg_id[cluster_determin_index][i])
            elif label_style == 'rand':
                rand_id = np.random.randint(low=0, high=len(cluster_main_seg_id[cluster_determin_index]))
                seg_label_id_list.append(cluster_main_seg_id[cluster_determin_index][rand_id])
            elif label_style == 'mainseg':
                main_seg_id = cluster_main_seg_id[cluster_determin_index]
                main_seg_point_num = cluster_main_seg_point_num[cluster_determin_index]
                for i in range(anno_num):
                    if i >= len(cluster_main_seg_id[cluster_determin_index]):
                        continue
                    repeat = True
                    while (repeat):
                        rand_index = np.random.randint(low=0, high=np.sum(main_seg_point_num))
                        for index in range(main_num):
                            if rand_index < np.sum(main_seg_point_num[:index+1]):
                                break
                        if cluster_main_seg_id[cluster_determin_index][index] not in seg_label_id_list:
                            seg_label_id_list.append(cluster_main_seg_id[cluster_determin_index][index])
                            repeat = False

            for j in range(len(clusters)):
                if j == cluster_determin_index:
                    continue
                if cluster_point_num[j] < 100:
                    continue
                if label_style == 'maxseg':
                    for i in range(anno_num):
                        if i < len(cluster_main_seg_id[j]):
                            seg_label_id_list.append(cluster_main_seg_id[j][i])
                elif label_style == 'rand':
                    rand_id = np.random.randint(low=0, high=len(cluster_main_seg_id[j]))
                    seg_label_id_list.append(cluster_main_seg_id[j][rand_id])
                elif label_style == 'mainseg':
                    main_seg_id = cluster_main_seg_id[j]
                    main_seg_point_num = cluster_main_seg_point_num[j]
                    for i in range(anno_num):
                        if i >= len(cluster_main_seg_id[j]):
                            continue
                        repeat = True
                        while (repeat):
                            rand_index = np.random.randint(low=0, high=np.sum(main_seg_point_num))
                            for index in range(main_num):
                                if rand_index < np.sum(main_seg_point_num[:index+1]):
                                    break
                            if cluster_main_seg_id[j][index] not in seg_label_id_list:
                                seg_label_id_list.append(cluster_main_seg_id[j][index])
                                repeat = False
    
    ins_weak_seg_np = np.ones(ins_labels.shape) * (-1)
    sem_weak_seg_np = np.ones(sem_labels.shape) * (-1)
    for seg_id in seg_label_id_list:
        indexs = np.where(seg_labels==seg_id)[0]
        ins_weak_seg_np[indexs] = ins_labels[indexs]
        sem_weak_seg_np[indexs] = sem_labels[indexs]
    ins_weak_seg_output = []
    sem_weak_seg_output = []
    for i in range(ins_weak_seg_np.shape[0]):
        ins_weak_seg_output.append(str(int(ins_weak_seg_np[i]))+'\n')
        sem_weak_seg_output.append(str(int(sem_weak_seg_np[i]))+'\n')
        
    '''
    weak semantic label:
        unlabeled:  -1, 0
        labeled:    1 ~ 40
    weak instance label:
        unlabeled:  -1, 0
        labeled:    1 ~ instance num
    '''

    # get output path
    if label_style == 'mainseg':
        label_style = label_style + '_' + str(main_num)
    if anno_num > 1:
        label_style = label_style + '_a' + str(anno_num)
    output_root = os.path.join('label', 'seg', label_style, 'raw', scene_name)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # save label
    with open(os.path.join(output_root, scene_name+'.ins.txt'), 'w') as f:
        f.writelines(ins_weak_seg_output)
    f.close()
    with open(os.path.join(output_root, scene_name+'.sem.txt'), 'w') as f:
        f.writelines(sem_weak_seg_output)
    f.close()

    labeled_seg_vert_num = np.where(ins_weak_seg_np != -1)[0].shape[0]
    vert_num = ins_weak_seg_np.shape[0]
    seg_num = seg_unique.shape[0]
    ins_num = ins_unique.shape[0]
    if ins_unique[0] == 0:
        ins_num = ins_num - 1
    return labeled_seg_vert_num, vert_num, len(seg_label_id_list), seg_num, ins_num


# Visualize labels on mesh
def visualize_labels(mesh_file, label_file, label_type, plydata=None, shuffle=False, adj_path=None, sem_labels=None):
    labels = np.array(load_labels(label_file))
    if sem_labels is not None:
        sem_labels = np.array(load_labels(sem_labels))
    if plydata is None:
        with open(mesh_file, 'rb') as f:
            plydata = PlyData.read(f)
        f.close()

    num_verts = plydata['vertex'].count
    if num_verts != len(labels):
        print('Loaded labels = ' + str(len(labels)) + 'vs mesh vertices = ' + str(num_verts))
        exit(1)
    
    if adj_path is not None:
        adj = np.array(torch.load(adj_path))
        adj_matrix = np.zeros((num_verts, num_verts), dtype=np.bool)
        for i in range(adj.shape[0]):
            adj_matrix[adj[i,0], adj[i,1]] = 1
            adj_matrix[adj[i,1], adj[i,0]] = 1
        weak_point_label_idxs = np.where(labels != -1)[0]
        for idx in weak_point_label_idxs:
            adj_point_idxs = np.where(adj_matrix[idx] == 1)[0]
            labels[adj_point_idxs] = labels[idx]
    
    if label_type == 'segment':
        labels_dict = np.unique(labels)
        if shuffle == True:
            random.shuffle(labels_dict)

    for i in range(num_verts):
        if labels[i] == -1:
            color = (255, 255, 255)
        elif (labels[i] == 0) and (label_type != 'segment'):
            color = (255, 255, 255)
        elif label_type == 'segment':
            color = colors[np.where(labels_dict==labels[i])[0][0]%num_colors+1]
            # color = colors[labels_dict[labels[i]]]
        elif label_type == 'instance':
            if sem_labels is not None and sem_labels[i].item() in [1,2]:
                color = (255, 255, 255)
            else:
                color = colors[(labels[i]-1)%num_colors+1]
        else:
            color = colors[labels[i]]
        plydata['vertex']['red'][i] = color[0]
        plydata['vertex']['green'][i] = color[1]
        plydata['vertex']['blue'][i] = color[2]

    output_file = '.'.join(label_file.split('.')[:-1])+'.ply'
    output_folder = os.path.join('/'.join(output_file.split('/')[:-1]), 'visualize')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join('/'.join(output_file.split('/')[:-1]), 'visualize', output_file.split('/')[-1])
    plydata.write(output_file)


# Visualize labels on mesh
def visualize_grouping_process(mesh_file, ins_label_file, seg_label_file, plydata=None, shuffle=True, seed=0):
    if plydata is None:
        with open(mesh_file, 'rb') as f:
            plydata = PlyData.read(f)
        f.close()
    num_verts = plydata['vertex'].count

    ins_labels = np.array(load_labels(ins_label_file))
    seg_labels = np.array(load_labels(seg_label_file))
    if (num_verts != ins_labels.shape[0]) or (num_verts != seg_labels.shape[0]):
        print('Loaded labels = ' + str(len(labels)) + 'vs mesh vertices = ' + str(num_verts))
        exit(1)
    
    
    ins_labels_dict = np.unique(ins_labels)[1:]
    seg_labels_dict = np.unique(seg_labels)
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(seg_labels_dict)

    for i in range(num_verts):
        if ins_labels[i] != -1:
            color = colors[np.where(ins_labels_dict==ins_labels[i])[0][0]%num_colors+1]
        else:
            if shuffle:
                np.random.seed(seed)
                color = colors[seg_labels[i]*np.random.randint(1,10)%num_colors+1]
            else:
                color = colors[seg_labels[i]%num_colors+1]
        plydata['vertex']['red'][i] = color[0]
        plydata['vertex']['green'][i] = color[1]
        plydata['vertex']['blue'][i] = color[2]

    output_file = '.'.join(seg_label_file.split('.')[:-1])+'.ply'
    output_folder = os.path.join('/'.join(output_file.split('/')[:-1]), 'visualize')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join('/'.join(output_file.split('/')[:-1]), 'visualize', output_file.split('/')[-1])
    plydata.write(output_file)


def cal_pairwise_distance(x, y):
    inner = -2*torch.matmul(x, y.transpose(1, 0))
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(1, 0)
    return pairwise_distance


def get_unmapper(x, y):
    idx_all = []
    point_num = 100000
    rep_num = x.shape[0] // point_num
    remainder_num = x.shape[0] % point_num
    if remainder_num != 0:
        rep_num += 1
    for i in range(rep_num):
        pairwise_distance = cal_pairwise_distance(x[i*point_num:(i+1)*point_num], y)
        idx = pairwise_distance.topk(k=1, dim=-1)[1].view(-1)
        idx_all.append(idx)
    idx_all = torch.cat(idx_all, dim=0)
    return idx_all


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


# Generate .pth file of point cloud for data-loader
def generate_pointcloud_pth(scene_path, item, num_points, plydata=None):
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]
    scene_name = os.path.split(scene_path)[-1]
    mesh_path = os.path.join(scene_path, scene_name + '_vh_clean_2.ply')

    if plydata is None:
        with open(mesh_path, 'rb') as f:
            plydata = PlyData.read(f)
        f.close()

    num_verts = plydata['vertex'].count
    pointcloud = np.zeros([num_verts, 6]) # [X, Y, Z, R, G, B]
    
    for i in range(num_verts):
        pointcloud[i,0] = plydata['vertex']['x'][i]
        pointcloud[i,1] = plydata['vertex']['y'][i]
        pointcloud[i,2] = plydata['vertex']['z'][i]
        pointcloud[i,3] = plydata['vertex']['red'][i]
        pointcloud[i,4] = plydata['vertex']['green'][i]
        pointcloud[i,5] = plydata['vertex']['blue'][i]

    # centring color
    pointcloud[:, 3:6] = pointcloud[:, 3:6] / 127.5 - 1

    # get output path
    output_root = os.path.join('data', 'resampled', scene_name)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_path1 = os.path.join(output_root, scene_name+'.pcl.pth')
    output_path2 = os.path.join(output_root, scene_name+'.info.pth')
    output_path3 = os.path.join(output_root, scene_name+'.map.pth')
    output_path4 = os.path.join(output_root, scene_name+'.unmap.pth')

    # save as torch tensor
    pointcloud = torch.FloatTensor(pointcloud)
    rep_num = num_points // pointcloud.shape[0]
    remainder_num = num_points % pointcloud.shape[0]
    if remainder_num > 0:
        # choice, _ = farthest_point_sampling(np.array(pointcloud[:, :3]), remainder_num)
        # index_remainder = torch.LongTensor(choice[0])
        index_remainder = torch.randperm(pointcloud.shape[0])[:remainder_num]
    else:
        index_remainder = torch.LongTensor([])
    if rep_num != 0:
        mapper = torch.cat([torch.arange(pointcloud.shape[0]).repeat(rep_num), index_remainder], dim=0)
    else:
        mapper = index_remainder
    pointcloud_sampled = pointcloud[mapper]
    info = torch.LongTensor([item])
    torch.save(pointcloud_sampled, output_path1)
    torch.save(info, output_path2)
    torch.save(mapper, output_path3)

    unmapper = torch.ones(pointcloud.shape[0], dtype=torch.long) * -100
    for i in range(mapper.shape[0]):
        unmapper[mapper[i]] = i
    unsampled_index = torch.where(unmapper == -100)[0]
    if unsampled_index.shape[0] != 0:
        unmapper[unsampled_index] = get_unmapper(pointcloud[unsampled_index,:3], pointcloud_sampled[:,:3])
    torch.save(unmapper, output_path4)


# Generate .pth of real label for data-loader
def generate_real_label_pth(scene_path):
    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]
    scene_name = os.path.split(scene_path)[-1]

    label_root = os.path.join('label', 'real', 'raw', scene_name)
    ins_path = os.path.join(label_root, scene_name + '.ins.txt')
    sem_path = os.path.join(label_root, scene_name + '.sem.txt')
    ins_labels = np.array(load_labels(ins_path))
    sem_labels = np.array(load_labels(sem_path))

    '''
    semantic label:
        unlabeled:  0                      
        labeled:    1 ~ 40
    instance label:
        unlabeled:  0                       
        labeled:    1 ~ instance num   
    '''

    num_verts = ins_labels.shape[0]
    label = np.zeros([num_verts, 2]) # [sem, ins]
    
    for i in range(num_verts):
        label[i,0] = sem_labels[i]
        label[i,1] = ins_labels[i]

    # get output path
    output_path = os.path.join(label_root, scene_name+'.label.pth')

    # save as torch tensor
    label = torch.LongTensor(label)
    torch.save(label, output_path)


# Generate .pth of weak label for data-loader
def generate_weak_label_pth(scene_name, label_style='manual'):
    label_root = os.path.join('label', 'seg', label_style, 'raw', scene_name)
    ins_weak_seg_path = os.path.join(label_root, scene_name + '.ins.txt')
    sem_weak_seg_path = os.path.join(label_root, scene_name + '.sem.txt')
    ins_weak_seg_labels = np.array(load_labels(ins_weak_seg_path))
    sem_weak_seg_labels = np.array(load_labels(sem_weak_seg_path))

    # change label starting index from 1 to 0
    for idx in np.where(ins_weak_seg_labels >= 0)[0]:
        ins_weak_seg_labels[idx] -= 1
        sem_weak_seg_labels[idx] -= 1

    '''
    weak semantic label:
        unlabeled:  -1                      
        labeled:    0 ~ 39
    weak instance label:
        unlabeled:  -1                      
        labeled:    0 ~ (instance num - 1)  
    '''

    num_verts = ins_weak_seg_labels.shape[0]
    weak_seg_labels = np.zeros([num_verts, 2])    # [sem_weak, ins_weak]
    weak_seg_labels[:,0] = sem_weak_seg_labels
    weak_seg_labels[:,1] = ins_weak_seg_labels

    # get output path
    output_root = os.path.join('label', 'seg', label_style, 'resampled', scene_name)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # save as torch tensor
    weak_seg_labels = torch.LongTensor(weak_seg_labels)
    mapper = torch.load(os.path.join('data', 'resampled', scene_name, scene_name+'.map.pth'))
    weak_seg_labels_sampled = weak_seg_labels[mapper]
    torch.save(weak_seg_labels_sampled, os.path.join(output_root, scene_name+'.label.pth'))


def get_adj_from_mesh(plydata, unmapper=None):
    if plydata is None:
        with open(mesh_path, 'rb') as f:
            plydata = PlyData.read(f)
        f.close()
    num_faces = plydata['face'].count
    adj = []
    for i in range(num_faces):
        face = plydata['face']['vertex_indices'][i]
        for idx in [[0,1], [0,2], [1,2]]:
            id1 = face[idx[0]]
            id2 = face[idx[1]]
            if id1 != id2:
                adj.append([id1, id2])
    adj = torch.LongTensor(adj)
    if unmapper is not None:
        adj_resampled = unmapper[adj.view(-1)].view(-1, 2)
        adj_resampled = torch.sort(adj_resampled, dim=-1)[0]
        adj_resampled = torch.unique(adj_resampled, dim=0)
    adj = torch.sort(adj, dim=-1)[0]
    adj = torch.unique(adj, dim=0)
    return adj, adj_resampled

# Generate .pth of mesh adjacency graph
def generate_mesh_adjcency_pth(scene_name, plydata=None):
    output_root1 = os.path.join('adj', 'mesh', 'raw', scene_name)
    output_root2 = os.path.join('adj', 'mesh', 'resampled', scene_name)
    adj_path = os.path.join(output_root1, scene_name+'.adj.pth')
    adj_resampled_path = os.path.join(output_root2, scene_name+'.adj.pth')
    if os.path.exists(adj_path) and os.path.exists(adj_resampled_path):
        return

    unmap_path = os.path.join('data', 'resampled', scene_name, scene_name+'.unmap.pth')
    unmapper = torch.load(unmap_path)
    adj, adj_resampled = get_adj_from_mesh(plydata, unmapper)
    if not os.path.exists(output_root1):
        os.makedirs(output_root1)
    torch.save(adj, adj_path)
    if not os.path.exists(output_root2):
        os.makedirs(output_root2)
    torch.save(adj_resampled, adj_resampled_path)


def get_adj_from_pointcloud(pointcloud, k=10):
    adj_resampled = []
    point_num = 100000
    rep_num = pointcloud.shape[0] // point_num
    remainder_num = pointcloud.shape[0] % point_num
    if remainder_num != 0:
        rep_num += 1
    for i in range(rep_num):
        pairwise_distance = cal_pairwise_distance(pointcloud[i*point_num:(i+1)*point_num], pointcloud)
        topk_idxs = pairwise_distance.topk(k=k+1, dim=-1)[1][:,1:].reshape(-1, 1)
        if (i+1)*point_num > pointcloud.shape[0]:
            end = pointcloud.shape[0]
        else:
            end = (i+1)*point_num
        point_ids = torch.arange(i*point_num, end).view(-1, 1).repeat(1, k).view(-1, 1)
        adj_resampled.append(torch.cat([point_ids, topk_idxs], dim=1))
    adj_resampled = torch.cat(adj_resampled, dim=0)
    adj_resampled = torch.LongTensor(adj_resampled)
    adj_resampled = torch.sort(adj_resampled, dim=-1)[0]
    adj_resampled = torch.unique(adj_resampled, dim=0)
    return adj_resampled
    