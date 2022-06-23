'''
Modified from SparseConvNet data preparation: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/prepare_data.py
'''

import os
import glob, plyfile, numpy as np, multiprocessing as mp, torch, json, argparse

import scannet_util

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i

parser = argparse.ArgumentParser()
parser.add_argument('--type', help='Style of pseudo label', default='manual')
opt = parser.parse_args()

split = 'train'
print('data split: {}'.format(split))
# files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
path = 'scannetv2_' + split + '.txt'
with open(path, 'r') as f:
    files = f.readlines()
f.close()
data_root = '/data1/antao/Documents/Datasets/ScanNet/scans/'

def f(fn):
    fn = fn[:-1]
    fn1 = os.path.join(data_root, fn, fn + '_vh_clean_2.ply')
    label_root = '/data1/antao/Documents/SegGroup/results/manual'
    fn2 = os.path.join(label_root, fn, 'epoch_last', 'final.sem.txt')
    fn3 = os.path.join(label_root, fn, 'epoch_last', 'final.ins.txt')
    print(fn1)

    f = plyfile.PlyData().read(fn1)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    with open(fn2, 'r') as f:
        f2 = f.readlines()
    f.close()
    for i in range(len(f2)):
        f2[i] = f2[i][:-1]
    sem_labels0 = np.array(f2).astype(int)
    sem_labels = remapper[sem_labels0]

    with open(fn3, 'r') as f:
        f3 = f.readlines()
    f.close()
    for i in range(len(f3)):
        f3[i] = f3[i][:-1]
    instance_labels0 = np.array(f3).astype(float)
    instance_labels = np.ones(sem_labels.shape[0]) * -100

    instance_id = 0
    instance_unique = np.sort(np.unique(instance_labels0))
    for ins in instance_unique:
        if ins == 0:
            continue
        indexs = np.where(instance_labels0 == ins)[0]
        if sem_labels0[indexs][0] in [1, 2]:
            continue
        instance_labels[indexs] = instance_id
        instance_id += 1
    
    torch.save((coords, colors, sem_labels, instance_labels), split + '_' + opt.type + '/' + fn+'_inst_nostuff.pth')
    print('Saving to ' + split + '_' + opt.type + '/' + fn+'_inst_nostuff.pth')

# for fn in files:
#     f(fn)

if not os.path.exists(split + '_' + opt.type):
    os.mkdir(split + '_' + opt.type)

p = mp.Pool(processes=mp.cpu_count())
p.map(f, files)
p.close()
p.join()

# f('scene0217_00\n')
