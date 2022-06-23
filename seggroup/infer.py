''' Inference script to generate pseudo labels

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

Required Inputs:
    --exp_name (str): Name of the experiment to resume.
    --label_style (int): Style of weak labels.
    --sem_infer (store_true): Infer pseudo labels for semantic segmentation.
    --ins_infer (store_true): Infer pseudo labels for instance segmentation.

Note: You need to choose either --sem_infer or --ins_infer.

Example Usage: 
    python infer.py --exp_name <your exp name> --label_style manual --ins_infer

'''

from __future__ import print_function
import os
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
import time
import argparse
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from data import ScanNet
from model import SegModel
from util import IOStream

SEM_VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
INS_VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEM_CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
INS_CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


def _init_(args):
    result_path = os.path.join('results', args.exp_name)
    if args.sem_infer:
        checkpoint_path = os.path.join('checkpoints', args.exp_name, 'sem_infer')
    elif args.ins_infer:
        checkpoint_path = os.path.join('checkpoints', args.exp_name, 'ins_infer')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    os.system('cp infer.py %s' % os.path.join(checkpoint_path, 'infer.py.backup'))
    os.system('cp model.py %s' % os.path.join(checkpoint_path, 'model.py.backup'))
    os.system('cp util.py %s' % os.path.join(checkpoint_path, 'util.py.backup'))
    os.system('cp data.py %s' % os.path.join(checkpoint_path, 'data.py.backup'))


def print_class_iou(IoU_sem_selected, IoU_ins_selected, acc_sem_selected_all, acc_ins_selected_all, io):
    outstr = 'Instance mIoU (18 classes): %.2f%%      Acc (18 classes): %.2f%%' % (np.nanmean(IoU_ins_selected)*100, acc_ins_selected_all*100)
    io.cprint(outstr)
    for i in range(18):
        outstr = '{:<16}{:<16}'.format(INS_CLASS_LABELS[i], '%.2f%%' % (IoU_ins_selected[i]*100))
        io.cprint(outstr)
    io.cprint('')

    outstr = 'Semantic mIoU (20 classes): %.2f%%      Acc (20 classes): %.2f%%' % (np.nanmean(IoU_sem_selected)*100, acc_sem_selected_all*100)
    io.cprint(outstr)
    for i in range(20):
        outstr = '{:<16}{:<16}'.format(SEM_CLASS_LABELS[i], '%.2f%%' % (IoU_sem_selected[i]*100))
        io.cprint(outstr)
    io.cprint('')


def main_worker(gpu, ngpus_per_node, args):
    io = IOStream(os.path.join('checkpoints', args.exp_name, 'run_infer.log'))
    args.gpu = gpu
    args.rank = gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:2344',
                                world_size=args.world_size, rank=args.rank)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = SegModel(exp_name=args.exp_name, cuda=args.cuda, visualize=args.visualize, 
                        sem_infer=args.sem_infer, ins_infer=args.ins_infer)
    model.to(device)
    if args.rank == 0:
        io.cprint('Network parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    args.batch_size = 1
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    train_dataset = ScanNet(label_style=args.label_style)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                        num_workers=args.workers, batch_size=args.batch_size, shuffle=False, 
                        pin_memory=True, sampler=train_sampler)
    # train_loader = DataLoader(train_dataset, 
    #                     num_workers=args.workers, batch_size=args.batch_size, shuffle=False, 
    #                     pin_memory=True)

    scene_list_path = os.path.join('dataset', 'scannet', 'scannetv2_train.txt')
    with open(scene_list_path, 'r') as f:
        scene_list = f.readlines()
    f.close()

    loc = 'cuda:{}'.format(args.gpu)
    checkpoint_path = 'checkpoints/%s/models/last.t7' % args.exp_name
    if not os.path.exists(checkpoint_path):
        if args.rank == 0:
            io.cprint('No checkpoint model, please make sure that you use right name in --exp_name')
        exit(1)
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    if args.rank == 0:
        io.cprint('Load model from ' + checkpoint_path)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])

    # train_sampler = None
    infer(train_loader, train_sampler, model, gpu, ngpus_per_node, args, io)


def infer(train_loader, train_sampler, model, gpu, ngpus_per_node, args, io):
    IoU_sem_best = 0
    IoU_ins_best = 0

    with torch.no_grad():
        if args.sem_infer:
            model.module.epoch = 'sem_infer'
        elif args.ins_infer:
            model.module.epoch = 'ins_infer'
        train_sampler.set_epoch(0)

        if args.rank == 0:
            I_sem_all = np.zeros(40)
            U_sem_all = np.zeros(40)
            I_ins_all = np.zeros(40)
            U_ins_all = np.zeros(40)
            acc_sem_all = 0
            acc_ins_all = 0
            acc_sem_selected_all = 0
            acc_ins_selected_all = 0


        for i, (data, weak_label, info) in enumerate(train_loader):
            data = data.cuda(args.gpu, non_blocking=True)
            weak_label = weak_label.cuda(args.gpu, non_blocking=True)
            IoU_sem, IoU_ins, acc = model(data, weak_label, info)

            dist.all_reduce(IoU_sem)
            dist.all_reduce(IoU_ins)
            dist.all_reduce(acc)

            if args.rank == 0:
                I_sem = np.array(IoU_sem[0,0].cpu())
                U_sem = np.array(IoU_sem[0,1].cpu())
                I_sem_all += I_sem
                U_sem_all += U_sem
                I_ins = np.array(IoU_ins[0,0].cpu())
                U_ins = np.array(IoU_ins[0,1].cpu())
                I_ins_all += I_ins
                U_ins_all += U_ins
                acc_sem_all += acc[0].item()
                acc_ins_all += acc[1].item()
                acc_sem_selected_all += acc[2].item()
                acc_ins_selected_all += acc[3].item()
                
                outstr = 'Infer(%04d/%04d)    Instance mIoU: %.2f%%    Semantic mIoU: %.2f%%    Instance Acc: %.2f%%    Semantic Acc: %.2f%%' \
                    % ((i+1)*ngpus_per_node, len(train_loader.dataset), 
                        np.nanmean(I_ins_all/U_ins_all)*100, np.nanmean(I_sem_all/U_sem_all)*100, 
                        acc_ins_all/((i+1)*ngpus_per_node)*100, acc_sem_all/((i+1)*ngpus_per_node)*100)
                io.cprint(outstr)

        if args.rank == 0:
            IoU_sem = I_sem_all/U_sem_all
            IoU_ins = I_ins_all/U_ins_all
            outstr = '==> Infer           Instance mIoU: %.2f%%    Semantic mIoU: %.2f%%    Instance Acc: %.2f%%    Semantic Acc: %.2f%%' \
                % (np.nanmean(I_ins_all/U_ins_all)*100, np.nanmean(I_sem_all/U_sem_all)*100, 
                    acc_ins_all/((i+1)*ngpus_per_node)*100, acc_sem_all/((i+1)*ngpus_per_node)*100)
            io.cprint(outstr)

            IoU_sem_selected = IoU_sem[SEM_VALID_CLASS_IDS-1]
            IoU_ins_selected = IoU_ins[INS_VALID_CLASS_IDS-1]
            io.cprint('')
            print_class_iou(IoU_sem_selected, IoU_ins_selected, \
                            acc_sem_selected_all/((i+1)*ngpus_per_node), acc_ins_selected_all/((i+1)*ngpus_per_node), io)


if __name__ == "__main__":
    # Inference settings
    parser = argparse.ArgumentParser(description='Pseudo Label Inference')
    parser.add_argument('-n', '--exp_name', required=True, type=str, default=None,
                        help='Name of the experiment to resume.')
    parser.add_argument('--label_style', type=str, default='manual',
                        help='Style of weak labels.')
    parser.add_argument('--sem_infer', action='store_true',
                        help='Infer pseudo labels for semantic segmentation.')
    parser.add_argument('--ins_infer', action='store_true',
                        help='Infer pseudo labels for instance segmentation.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8).')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Don\'t use CUDA for training.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Visualize results.')
    args = parser.parse_args()

    if ((args.sem_infer==True) and (args.ins_infer==True)) or ((args.sem_infer==False) and (args.ins_infer==False)):
        print('Please choose either \'--sem_infer\' or \'--ins_infer\'')
        exit(1)

    _init_(args)

    io = IOStream(os.path.join('checkpoints', args.exp_name, 'run_infer.log'))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    np.random.seed(1)
    if args.cuda:
        io.cprint(
            "Let's use " + str(torch.cuda.device_count()) + " GPUs!")
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    ngpus_per_node = torch.cuda.device_count()
    args.batch_size = ngpus_per_node
    args.world_size = ngpus_per_node
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # main_worker(gpu=0, ngpus_per_node=ngpus_per_node, args=args)

