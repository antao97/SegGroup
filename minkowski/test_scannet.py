# TODO: this file can reproduce 72.22% mIoU on ScanNetV2 val set

import os
import argparse
import numpy as np
from plyfile import PlyData

import torch
import MinkowskiEngine as ME
import utils

from models.res16unet import Res16UNet34C

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str,
                    default='/data3/antao/Documents/SpatioTemporalSegmentation/pretrained/'
                            'MinkUNet34C-train-conv1-5.pth')
parser.add_argument('--data_path', type=str,
                    default='/data3/antao/Documents/Datasets/'
                            'ScanNet_processed/train')
parser.add_argument('--bn_momentum', type=float, default=0.05)
parser.add_argument('--voxel_size', type=float, default=0.02)
parser.add_argument('--conv1_kernel_size', type=int, default=5)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])

VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                   10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
VALID_CLASS_NAMES = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                    'counter', 'desk', 'curtain', 'refrigerator', 'showercurtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
ignore_class_ids = tuple(set(range(41)) - set(VALID_CLASS_IDS))


def convert_label_scannet(label):
    for ignore_lbl in ignore_class_ids:
        if ignore_lbl in label:
            label[label == ignore_lbl] = 255
    for i, lbl in enumerate(VALID_CLASS_IDS):
        if lbl in label:
            label[label == lbl] = i
    return label


def load_file(file_name, voxel_size):
    plydata = PlyData.read(file_name+'.ply')
    data = plydata.elements[0].data
    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    colors = np.array([data['red'], data['green'],
                       data['blue']], dtype=np.float32).T / 255.
    labels = np.array(data['label'], dtype=np.int32)

    feats = colors - 0.5

    idx, inverse_idx, quan_coords, quan_feats = utils.sparse_quantize(
        coords, feats, None, return_index=True,
        return_inverse=True, quantization_size=voxel_size)

    return inverse_idx, quan_coords, quan_feats, labels


def generate_input_sparse_tensor(file_name, voxel_size=0.02):
    # Create a batch, this process is done in a data loader during training in parallel.
    batch = [load_file(file_name, voxel_size)]
    inverse_idx, coordinates_, featrues_, labels = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(
        coordinates_, featrues_, None)

    # Normalize features and create a sparse tensor
    return inverse_idx, coordinates, features.float(), labels[0]


def save_prediction(save_path, split, room_name, pred, prob):
    pred = np.array(VALID_CLASS_IDS, dtype=np.int32)[pred]
    np.savetxt(os.path.join(config.save_path, split, 'pred', room_name+'.txt'), pred, fmt='%d')
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=6)
    np.savetxt(os.path.join(config.save_path, split, 'prob', room_name+'.txt'), prob, fmt='%.6f')
    return


if __name__ == '__main__':
    config = parser.parse_args()
    # download(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.save_path is not None:
        if not os.path.exists(os.path.join(config.save_path, config.split)):
            os.makedirs(os.path.join(config.save_path, config.split, 'pred'))
            os.makedirs(os.path.join(config.save_path, config.split, 'prob'))

    # Define a model and load the weights
    model = Res16UNet34C(3, 20, config).to(device)
    model_dict = torch.load(config.weights)
    model.load_state_dict(model_dict['state_dict'])
    model.eval()

    # test
    label_all, pred_all = np.array([]), np.array([])
    if config.split == 'val':
        with open(os.path.join(config.data_path, 'scannetv2_val.txt'), 'r') as f:
            all_rooms = f.readlines()
    elif config.split == 'train':
        with open(os.path.join(config.data_path, 'scannetv2_train.txt'), 'r') as f:
            all_rooms = f.readlines()
    else:
        with open(os.path.join(config.data_path, 'scannetv2_test.txt'), 'r') as f:
            all_rooms = f.readlines()
    all_rooms = [room[:-1] for room in all_rooms]

    room_num = len(all_rooms)
    num_classes = len(VALID_CLASS_IDS)
    print('ScanNet num_classes: {}'.format(num_classes))
    for idx, room_name in enumerate(all_rooms):
        with torch.no_grad():
            data = os.path.join(config.data_path, room_name)
            inverse_idx, coordinates, features, label = \
                generate_input_sparse_tensor(
                    data,
                    voxel_size=config.voxel_size)
            label = convert_label_scannet(label)
            
            # Feed-forward pass and get the prediction
            sinput = ME.SparseTensor(features, coords=coordinates).to(device)
            soutput = model(sinput)
            pred = soutput.F.max(1)[1].cpu().numpy()
            prob = torch.nn.functional.softmax(soutput.F, dim=1).cpu().numpy()

            pred = pred[inverse_idx]
            prob = prob[inverse_idx]
            
            if config.save_path is not None:
                save_prediction(config.save_path, config.split, room_name, pred, prob)

            if config.split != 'test':
                intersection, union, target = utils.intersectionAndUnion(
                    pred, label, num_classes, 255)
                mIoU = np.nanmean(intersection / union)
                print('Room {}/{} mIoU {:.4F}'.format(idx, room_num, mIoU))

                # save results
                pred_all = np.hstack([pred_all, pred]) if \
                    pred_all.size else pred
                label_all = np.hstack([label_all, label]) if \
                    label_all.size else label
            else:
                print('Room {}/{}'.format(idx, room_num))

            torch.cuda.empty_cache()

    if config.split != 'test':
        intersection, union, target = \
            utils.intersectionAndUnion(
                pred_all, label_all, num_classes, 255)
        iou_class = intersection / (union + 1e-10)
        accuracy_class = intersection / (target + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection) / (sum(target) + 1e-10)
        print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.
              format(mIoU, mAcc, allAcc))
        for i in range(num_classes):
            print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.
                  format(i, iou_class[i], accuracy_class[i], VALID_CLASS_NAMES[i]))
    else:
        print('Done!')
