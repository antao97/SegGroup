''' Training script of the SegGroup model

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

Required Inputs:
    --label_style (str): Style of weak labels.

Important Optional Inputs:
    --exp_name (str): Name of the experiment (default is <date>_<time>).
    --resume (store_true): Resume training from the last checkpoint.
    --epochs (int): Number of the episode to train.

Example Usage: 
    python train.py --label_style manual 

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


def _init_():
    result_path = os.path.join('results', args.exp_name)
    checkpoint_path = os.path.join('checkpoints', args.exp_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(checkpoint_path, 'models')):
        os.makedirs(os.path.join(checkpoint_path, 'models'))
    os.system('cp train.py %s' % os.path.join(checkpoint_path, 'train.py.backup'))
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
    io = IOStream(os.path.join('checkpoints', args.exp_name, 'run.log'))
    args.gpu = gpu
    args.rank = gpu
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456',
                                world_size=args.world_size, rank=args.rank)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = SegModel(exp_name=args.exp_name, cuda=args.cuda, visualize=args.visualize)
    model.to(device)
    if args.rank == 0:
        io.cprint('Network parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))
    
    args.batch_size = 1
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    if args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

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

    if args.resume:
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
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        args.start_epoch = 0


    # train_sampler = None
    train(train_loader, train_sampler, model, optimizer, gpu, ngpus_per_node, args, io)


def train(train_loader, train_sampler, model, optimizer, gpu, ngpus_per_node, args, io):
    IoU_sem_best = 0
    IoU_ins_best = 0

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        if epoch == (args.epochs - 1):
            model.module.epoch = 'last'
        else:
            model.module.epoch = str(epoch+1)
        train_sampler.set_epoch(epoch)

        if args.rank == 0:
            train_loss = 0
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
            loss_raw, IoU_sem, IoU_ins, acc = model(data, weak_label, info)

            loss_sum = torch.sum(loss_raw[:,0])
            loss_num = torch.sum(loss_raw[:,1])
            loss = loss_sum / loss_num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dist.all_reduce(loss)
            dist.all_reduce(IoU_sem)
            dist.all_reduce(IoU_ins)
            dist.all_reduce(acc)

            if args.rank == 0:
                train_loss += loss.item()
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
                
                outstr = 'Epoch[%d/%d](%04d/%04d)    Loss: %.6f    Instance mIoU: %.2f%%    Semantic mIoU: %.2f%%    Instance Acc: %.2f%%    Semantic Acc: %.2f%%' \
                    % (epoch+1, args.epochs, 
                        (i+1)*ngpus_per_node, len(train_loader.dataset), 
                        train_loss/((i+1)*ngpus_per_node), 
                        np.nanmean(I_ins_all/U_ins_all)*100, np.nanmean(I_sem_all/U_sem_all)*100, 
                        acc_ins_all/((i+1)*ngpus_per_node)*100, acc_sem_all/((i+1)*ngpus_per_node)*100)
                io.cprint(outstr)

        if args.rank == 0:
            IoU_sem = I_sem_all/U_sem_all
            IoU_ins = I_ins_all/U_ins_all
            outstr = '==> Epoch[%d/%d]           Loss: %.6f    Instance mIoU: %.2f%%    Semantic mIoU: %.2f%%    Instance Acc: %.2f%%    Semantic Acc: %.2f%%' \
                % (epoch+1, args.epochs, 
                    train_loss/((i+1)*ngpus_per_node), 
                    np.nanmean(I_ins_all/U_ins_all)*100, np.nanmean(I_sem_all/U_sem_all)*100, 
                    acc_ins_all/((i+1)*ngpus_per_node)*100, acc_sem_all/((i+1)*ngpus_per_node)*100)
            io.cprint(outstr)

            IoU_sem_selected = IoU_sem[SEM_VALID_CLASS_IDS-1]
            IoU_ins_selected = IoU_ins[INS_VALID_CLASS_IDS-1]
            io.cprint('')
            print_class_iou(IoU_sem_selected, IoU_ins_selected, \
                            acc_sem_selected_all/((i+1)*ngpus_per_node), acc_ins_selected_all/((i+1)*ngpus_per_node), io)
            
            checkpoint = {'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict()}
            torch.save(checkpoint, 'checkpoints/%s/models/epoch_%d.t7' % (args.exp_name, epoch+1))
            torch.save(checkpoint, 'checkpoints/%s/models/last.t7' % args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point-Level Pseudo Label Generation')
    parser.add_argument('-n', '--exp_name', type=str, default=None,
                        help='Name of the experiment (default is to use date_time).')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume training from the last checkpoint.')
    parser.add_argument('--epochs', type=int, default=6, 
                        help='Number of the episode to train.')
    parser.add_argument('--label_style', type=str, default='manual',
                        help='Style of weak labels.')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='Number of data loading workers (default: 8).')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='Learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Don\'t use CUDA for training.')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='Random seed (default: 1)')
    parser.add_argument('-v', '--visualize', action='store_true', 
                        help='Visualize results.')
    args = parser.parse_args()

    if args.resume and args.exp_name is None:
        print('Please choose a specific experiment to resume by using \'--exp_name\'')
        exit(1)
    
    if args.exp_name is None:
        args.exp_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) 
    # args.exp_name = 'exp_test'

    _init_()

    io = IOStream(os.path.join('checkpoints', args.exp_name, 'run.log'))
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

