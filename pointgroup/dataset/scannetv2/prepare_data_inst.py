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
parser.add_argument('--data_split', help='data split (train / val / test)', default='train')
opt = parser.parse_args()

split = opt.data_split
print('data split: {}'.format(split))
# files = sorted(glob.glob(split + '/*_vh_clean_2.ply'))
path = 'scannetv2_' + split + '.txt'
with open(path, 'r') as f:
    files = f.readlines()
f.close()
if split != 'test':
    data_root = '/data1/antao/Documents/Datasets/ScanNet/scans/'
else:
    data_root = '/data1/antao/Documents/Datasets/ScanNet/scans_test/'
# if opt.data_split != 'test':
#     files2 = sorted(glob.glob(split + '/*_vh_clean_2.labels.ply'))
#     files3 = sorted(glob.glob(split + '/*_vh_clean_2.0.010000.segs.json'))
#     files4 = sorted(glob.glob(split + '/*[0-9].aggregation.json'))
#     assert len(files) == len(files2)
#     assert len(files) == len(files3)
#     assert len(files) == len(files4), "{} {}".format(len(files), len(files4))

def f_test(fn):
    fn = fn[:-1]
    fn1 = os.path.join(data_root, fn, fn + '_vh_clean_2.ply')
    print(fn1)

    f = plyfile.PlyData().read(fn1)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    torch.save((coords, colors), split + '/' + fn + '_inst_nostuff.pth')
    print('Saving to ' + split + '/' + fn + '_inst_nostuff.pth')


def f(fn):
    fn = fn[:-1]
    fn1 = os.path.join(data_root, fn, fn + '_vh_clean_2.ply')
    fn2 = os.path.join(data_root, fn, fn + '_vh_clean_2.labels.ply')
    fn3 = os.path.join(data_root, fn, fn + '_vh_clean_2.0.010000.segs.json')
    fn4 = os.path.join(data_root, fn, fn + '.aggregation.json')
    print(fn1)

    f = plyfile.PlyData().read(fn1)
    points = np.array([list(x) for x in f.elements[0]])
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1

    f2 = plyfile.PlyData().read(fn2)
    sem_labels = remapper[np.array(f2.elements[0]['label'])]

    with open(fn3) as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    segid_to_pointid = {}
    for i in range(len(seg)):
        if seg[i] not in segid_to_pointid:
            segid_to_pointid[seg[i]] = []
        segid_to_pointid[seg[i]].append(i)

    instance_segids = []
    labels = []
    with open(fn4) as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            if scannet_util.g_raw2scannetv2[x['label']] != 'wall' and scannet_util.g_raw2scannetv2[x['label']] != 'floor':
                instance_segids.append(x['segments'])
                labels.append(x['label'])
                assert(x['label'] in scannet_util.g_raw2scannetv2.keys())
    if(fn == 'scene0217_00' and instance_segids[0] == instance_segids[int(len(instance_segids) / 2)]):
        instance_segids = instance_segids[: int(len(instance_segids) / 2)]
    check = []
    for i in range(len(instance_segids)): check += instance_segids[i]
    assert len(np.unique(check)) == len(check)

    instance_labels = np.ones(sem_labels.shape[0]) * -100
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_labels[pointids] = i
        assert(len(np.unique(sem_labels[pointids])) == 1)

    torch.save((coords, colors, sem_labels, instance_labels), split + '/' + fn+'_inst_nostuff.pth')
    print('Saving to ' + split + '/' + fn+'_inst_nostuff.pth')

# for fn in files:
#     f(fn)

p = mp.Pool(processes=mp.cpu_count())
if opt.data_split == 'test':
    p.map(f_test, files)
else:
    p.map(f, files)
p.close()
p.join()

# f('scene0217_00\n')
