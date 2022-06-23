''' Data preparation for the ScanNet dataset

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/10/12

Required Inputs:
    --data_root (str): Root path for the ScanNet dataset (e.g. the path includes 'scans' and 'scans_test' folder).
  
Example Usage: 
    python prepare_data.py --data_root <scannet path>

'''

import os
import glob
import argparse
import multiprocessing as mp
from plyfile import PlyData

from util import generate_seg_labels_and_ds_set, generate_real_labels, generate_pointcloud_pth, \
                    generate_real_label_pth, visualize_labels

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, type=str, 
                    help='Root path for dataset (e.g. the path includes scans and scans_test folder).')
parser.add_argument('--num_points', type=int, default=150000,
                    help='Number of points to sample.')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize labels on mesh.')
opt = parser.parse_args()


def main(para):
    scene_path = para[0]
    item = para[1]
    scene_num = para[2]
    scene_name = scene_path.split('/')[-1]
    print('[%04d/%d] Dealing with %s...' % (item, scene_num, scene_name))

    # Load mesh
    mesh_path = os.path.join(scene_path, scene_name+'_vh_clean_2.ply')
    with open(mesh_path, 'rb') as f:
        plydata = PlyData.read(f)
    f.close()
    
    # Generate real instance and semantic labels in .txt
    generate_real_labels(scene_path)

    # Visualize semantic and instance labels on mesh
    if opt.visualize:
        sem_label_path = os.path.join('label', 'real', 'raw', scene_name, scene_name+'.sem.txt')
        ins_label_path = os.path.join('label', 'real', 'raw', scene_name, scene_name+'.ins.txt')
        visualize_labels(mesh_path, sem_label_path, 'semantic', plydata)
        visualize_labels(mesh_path, ins_label_path, 'instance', plydata)

    # Generate .pth file for data-loader
    generate_pointcloud_pth(scene_path, item, opt.num_points, plydata)
    generate_real_label_pth(scene_path)

    # Generate segment labels in .txt and disjoint-set in .json
    generate_seg_labels_and_ds_set(scene_path)

    # Visualize segment labels on mesh
    if opt.visualize:
        seg_label_path = os.path.join('label', 'real', 'raw', scene_name, scene_name+'.seg.txt')
        visualize_labels(mesh_path, seg_label_path, 'segment', plydata)

    print('[%04d/%d] Finish %s!' % (item, scene_num, scene_name))


path = 'scannetv2_train.txt'
with open(path, 'r') as f:
    files = f.readlines()
f.close()
scene_paths = [os.path.join(opt.data_root, 'scans', f[:-1]) for f in files]
paras = [(scene_paths[i], i, len(scene_paths)) for i in range(len(scene_paths))]

# Use multiprocessing
p = mp.Pool(processes=4)
p.map(main, paras)
p.close()
p.join()

## Disable multiprocessing
# for para in paras:
#     main(para)

print('\nFinish all!\n')
