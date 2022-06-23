''' Weak label preparation for the ScanNet dataset

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/10/12

Required Inputs:
    --data_root (str): Root path for the ScanNet dataset (e.g. the path includes 'scans' and 'scans_test' folder).
    --label_style (str): Style to generate weak seg labels [manual, maxseg, mainseg, rand].
    
Important Optional Inputs:
    --manual_label_path (str): Path for manual annotated weak labels (only for manual).
    --main_num (int): Number of main segments to consider (only for mainseg).
    --anno_num (int): Number of annotations per instance (only for maxseg and mainseg).
      
Example Usage: 
    python prepare_weak_label.py --data_root <scannet path> --label_style manual --manual_label_path <label path>

'''

import os
import glob
import argparse
import multiprocessing as mp
from plyfile import PlyData

from util import generate_weak_labels, generate_weak_label_pth, \
                 visualize_labels, generate_mesh_adjcency_pth

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', required=True, type=int, 
                    help='Root path for dataset (e.g. the path includes scans and scans_test folder).')
parser.add_argument('--label_style', required=True, default='manual', 
                    choices=['manual', 'maxseg', 'mainseg', 'rand'],
                    help='Style to genearte weak seg labels.')
parser.add_argument('--manual_label_path', type=str, default=None, 
                    help='Path for manual annotated weak labels (only for manual).')
parser.add_argument('--main_num', type=int, default=-1, 
                    help='Number of main segments to consider (only for mainseg).')
parser.add_argument('--anno_num', type=int, default=1, 
                    help='Number of annotations per instance (only for maxseg and mainseg).')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize labels on mesh.')
opt = parser.parse_args()

if (opt.label_style == 'manual') and (opt.manual_label_path is None):
    print('Please give the path for manual weak labels!')
    exit(1)

if (opt.label_style == 'mainseg') and (opt.anno_num > opt.main_num):
    print('Please choose \'--anno_num\' smaller than \'--main_num\' when using \'--label_style=mainseg\'')
    exit(1)

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

    generate_mesh_adjcency_pth(scene_name, plydata)

    # Generate weak labels in .txt
    labeled_point, all_point, labeled_seg, all_seg, ins_num = generate_weak_labels(scene_path, plydata, 
                                                            opt.label_style, opt.manual_label_path, opt.main_num, opt.anno_num)

    label_style = opt.label_style
    if opt.label_style == 'mainseg':
        label_style = opt.label_style + '_' + str(opt.main_num)
    if opt.anno_num > 1:
        label_style = label_style + '_a' + str(opt.anno_num)

    # Visualize generated weak segment labels
    if opt.visualize:
        sem_weak_seg_label_path = os.path.join('label', 'seg', label_style, 'raw', scene_name, scene_name+'.sem.txt')
        ins_weak_seg_label_path = os.path.join('label', 'seg', label_style, 'raw', scene_name, scene_name+'.ins.txt')
        visualize_labels(mesh_path, sem_weak_seg_label_path, 'semantic', plydata)
        visualize_labels(mesh_path, ins_weak_seg_label_path, 'instance', plydata)

    # Generate .pth of weak label for data-loader
    generate_weak_label_pth(scene_name, label_style)

    print('[%04d/%d] Finish %s!' % (item, scene_num, scene_name))
    return labeled_point, all_point, labeled_seg, all_seg, ins_num


path = 'scannetv2_train.txt'
with open(path, 'r') as f:
    files = f.readlines()
f.close()
scene_paths = [os.path.join(opt.data_root, 'scans', f[:-1]) for f in files]
paras = [(scene_paths[i], i, len(scene_paths)) for i in range(len(scene_paths))]

# Use multiprocessing
# p = mp.Pool(processes=mp.cpu_count())
p = mp.Pool(processes=8)
ret = p.map(main, paras)
p.close()
p.join()

## Disable multiprocessing
# ret = []
# for para in paras:
#     r = main(para)
#     ret.append(r)

labeled_point_count = 0
all_point_count = 0
labeled_seg_count = 0
all_seg_count = 0
ins_count = 0
for i in range(len(ret)):
    labeled_point_count += ret[i][0]
    all_point_count += ret[i][1]
    labeled_seg_count += ret[i][2]
    all_seg_count += ret[i][3]
    ins_count += ret[i][4]

print('\nAverage labeled points of weak label in all points: %.2f%%' % (labeled_point_count/all_point_count*100))
print('Average labeled segments of weak label in all segments: %.2f%%' % (labeled_seg_count/all_seg_count*100))
print('Average real labeled points in all points: %.4f%%' % (labeled_seg_count/all_point_count*100))
print('Average instance number per scene: %.2f' % (ins_count/1201))
print('Average labeled segment number per scene: %.2f' % (labeled_seg_count/1201))

print('\nFinish all!\n')
