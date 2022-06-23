''' Dataloader of the ScanNet dataset

SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation

Author: An Tao
Email: ta19@mails.tsinghua.edu.cn
Date: 2020/7/5

'''

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class ScanNet(Dataset):
    def __init__(self, label_style='manual'):
        self.label_style = label_style
        self.data_root = os.path.join('dataset', 'scannet')

        scene_list_path = os.path.join(self.data_root, 'scannetv2_train.txt')
        with open(scene_list_path, 'r') as f:
            self.scene_list = f.readlines()
        f.close()
        
    def __getitem__(self, item):
        scene_name = self.scene_list[item][:-1]
        pcl_path = os.path.join(self.data_root, 'data', 'resampled', scene_name, scene_name+'.pcl.pth')
        weak_label_path = os.path.join(self.data_root, 'label', 'seg', self.label_style, 'resampled', scene_name, scene_name+'.label.pth')
        info_path = os.path.join(self.data_root, 'data', 'resampled', scene_name, scene_name+'.info.pth')

        data = torch.load(pcl_path)
        # data[:,:2] -= data[:,:2].mean(0)
        weak_label = torch.load(weak_label_path)
        info = torch.load(info_path)
        return data, weak_label, info

    def __len__(self):
        return len(self.scene_list)


if __name__ == '__main__':
    train = ScanNet()
    data, label, info = train[0]
    print(data.shape)
    print(label.shape)
    print(info.shape)
