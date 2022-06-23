''' Evaluation script for already saved pseudo labels

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/10/12

Required Inputs:
    --exp_name (str): Name of the experiment to evaluate.
    --layer (str): Which layer of pseudo labels to evaluate ['1', '2', '3', '4', 'final'].
    --stage (str): Which stage of pseudo labels to evaluate ['epoch_1', ... , 'epoch_last', 'ins_infer', 'sem_infer'].

Note: The evaluated pseudo labels need to be already saved with TXT format.
  
Example Usage: 
    python evaluate.py --exp_name <your exp name> --layer final --epoch last

'''

import os
import torch
import numpy as np
import multiprocessing as mp
import sklearn.metrics as metrics

from dataset.scannet.util import load_labels
from main_scannet import print_class_iou

SEM_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
INS_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEM_CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
INS_CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


def evaluate(scene_name):
    scene_name = scene_name[:-1]

    ### Evaluation
    # Prepare for evaluation
    real_label_root = os.path.join('dataset', 'scannet', 'label', 'real', 'raw', scene_name)
    real_label_path = os.path.join(real_label_root, scene_name+'.label.pth')
    real_label = torch.load(real_label_path) 
    sem_true = np.array(real_label[:,0])
    ins_true = np.array(real_label[:,1])

    pred_label_root = os.path.join('results', args.exp_name, scene_name, args.stage)
    if args.layer == 'final':
        pred_sem_label_path = os.path.join(pred_label_root, 'final.sem.txt')
        pred_ins_label_path = os.path.join(pred_label_root, 'final.ins.txt')
    else:
        pred_sem_label_path = os.path.join(pred_label_root, 'layer_'+args.layer+'.sem.txt')
        pred_ins_label_path = os.path.join(pred_label_root, 'layer_'+args.layer+'.ins.txt')
    sem_pred = np.array(load_labels(pred_sem_label_path))
    ins_pred = np.array(load_labels(pred_ins_label_path))

    valid_idxs = np.where(sem_true != 0)[0]
    sem_true = sem_true[valid_idxs]
    ins_true = ins_true[valid_idxs]
    sem_pred = sem_pred[valid_idxs]
    ins_pred = ins_pred[valid_idxs]

    # Calculate semantic mIoU
    I_sem = np.zeros([1, 40])
    U_sem = np.zeros([1, 40])
    for idx in range(40):
        sem = idx + 1
        I_sem[0, idx] += np.sum(np.logical_and(sem_pred == sem, sem_true == sem))
        U_sem[0, idx] += np.sum(np.logical_or(sem_pred == sem, sem_true == sem))
    IoU_sem = np.concatenate([I_sem, U_sem], axis=0)     # [2, 40]

    # Calculate instance mIoU
    I_ins = np.zeros([1, 40])
    U_ins = np.zeros([1, 40])
    ins_unique = np.unique(ins_pred)
    for ins in ins_unique:
        sem = sem_pred[np.where(ins_pred == ins)[0][0]]
        idx = sem - 1
        I_ins[0, idx] += np.sum(np.logical_and(ins_pred == ins, ins_true == ins))
        U_ins[0, idx] += np.sum(np.logical_or(ins_pred == ins, ins_true == ins))
    IoU_ins = np.concatenate([I_ins, U_ins], axis=0)     # [2, 40]

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
    acc = np.array([acc_sem, acc_ins, acc_sem_selected, acc_ins_selected])  # 4

    return IoU_sem, IoU_ins, acc


# Evaluation settings
parser = argparse.ArgumentParser(description='Pseudo Label Evaluation')
parser.add_argument('-n', '--exp_name', required=True, type=str, default=None,
                    help='Name of the experiment to evaluate.')
parser.add_argument('--layer', type=str, default='final', 
                    choices=['1', '2', '3', '4', 'final'],
                    help='Which layer of pseudo labels to evaluate.')
parser.add_argument('--stage', type=str, default='last', 
                    help='Which stage of pseudo labels to evaluate. \
                        ['epoch_1', ... , 'epoch_last', 'ins_infer', 'sem_infer']')
args = parser.parse_args()

I_sem_all = np.zeros(40)
U_sem_all = np.zeros(40)
I_ins_all = np.zeros(40)
U_ins_all = np.zeros(40)
acc_sem_all = 0
acc_ins_all = 0
acc_sem_selected_all = 0
acc_ins_selected_all = 0

path = 'dataset/scannet/scannetv2_train.txt'
with open(path, 'r') as f:
    files = f.readlines()
f.close()

# Use multiprocessing
p = mp.Pool(processes=mp.cpu_count())
rets = p.map(evaluate, files)
p.close()
p.join()
for ret in rets:
    IoU_sem, IoU_ins, acc = ret[0], ret[1], ret[2]
    I_sem_all += IoU_sem[0]
    U_sem_all += IoU_sem[1]
    I_ins_all += IoU_ins[0]
    U_ins_all += IoU_ins[1]
    acc_sem_all += acc[0].item()
    acc_ins_all += acc[1].item()
    acc_sem_selected_all += acc[2].item()
    acc_ins_selected_all += acc[3].item()

## Disable multiprocessing
# count = 1
# for fn in files:
#     print('\r'+str(count))
#     count += 1

#     IoU_sem, IoU_ins, acc = evaluate(fn)
#     I_sem_all += IoU_sem[0]
#     U_sem_all += IoU_sem[1]
#     I_ins_all += IoU_ins[0]
#     U_ins_all += IoU_ins[1]
#     acc_sem_all += acc[0].item()
#     acc_ins_all += acc[1].item()
#     acc_sem_selected_all += acc[2].item()
#     acc_ins_selected_all += acc[3].item()

# Calculate mIoU
IoU_sem = I_sem_all/U_sem_all
IoU_ins = I_ins_all/U_ins_all
outstr = 'Instance mIoU: %.2f%%    Semantic mIoU: %.2f%%    Instance Acc: %.2f%%    Semantic Acc: %.2f%%' \
    % (np.nanmean(I_ins_all/U_ins_all)*100, np.nanmean(I_sem_all/U_sem_all)*100, 
        acc_ins_all/len(rets)*100, acc_sem_all/len(rets)*100)
print(outstr)

IoU_sem_selected = IoU_sem[SEM_VALID_CLASS_IDS-1]
IoU_ins_selected = IoU_ins[INS_VALID_CLASS_IDS-1]
outstr = '\nInstance mIoU (18 classes): %.2f%%      Acc (18 classes): %.2f%%' % (np.nanmean(IoU_ins_selected)*100, acc_ins_selected_all/len(rets)*100)
print(outstr)
for i in range(18):
    outstr = '{:<16}{:<16}'.format(INS_CLASS_LABELS[i], '%.2f%%' % (IoU_ins_selected[i]*100))
    print(outstr)

outstr = '\nSemantic mIoU (20 classes): %.2f%%      Acc (20 classes): %.2f%%' % (np.nanmean(IoU_sem_selected)*100, acc_sem_selected_all/len(rets)*100)
print(outstr)
for i in range(20):
    outstr = '{:<16}{:<16}'.format(SEM_CLASS_LABELS[i], '%.2f%%' % (IoU_sem_selected[i]*100))
    print(outstr)
