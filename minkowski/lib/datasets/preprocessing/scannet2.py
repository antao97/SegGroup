from pathlib import Path

import numpy as np
from lib.pc_utils import read_plyfile, save_point_cloud
from concurrent.futures import ProcessPoolExecutor

EXP_NAME = 'manual'
LABEL_NAME = 'infer_sem/layer_2.sem.txt'

SCANNET_RAW_PATH2 = Path('/data3/antao/Documents/Datasets/ScanNet/scans/')
LABEL_PATH = Path('/data1/antao/Documents/SegGroup/results/') / EXP_NAME
SCANNET_OUT_PATH2 = Path('/data3/antao/Documents/Datasets/ScanNet_processed/') / EXP_NAME
POINTCLOUD_FILE = '_vh_clean_2.ply'

print('start preprocess')
# Preprocess data.


def handle_process(path):
  scene = path.split(',')[0]
  label_f = Path(path.split(',')[1])
  phase_out_path = Path(path.split(',')[2])
  pointcloud_f = SCANNET_RAW_PATH2 / scene / (scene + POINTCLOUD_FILE)
  pointcloud = read_plyfile(pointcloud_f)
  # Make sure alpha value is meaningless.
  assert np.unique(pointcloud[:, -1]).size == 1
  # Load label file.
  with open(label_f, 'r') as f:
    label = f.readlines()
  f.close()
  for i in range(len(label)):
    label[i] = label[i][:-1]
  label = np.array(label).astype('float32')
  label[np.where(label==-1)[0]] = 0
  # Sanity check that the pointcloud and its label has same vertices.
  assert pointcloud.shape[0] == label.shape[0]
  out_f = phase_out_path / (scene + '.ply')
  processed = np.hstack((pointcloud[:, :6], np.array([label]).T))
  save_point_cloud(processed, out_f, with_label=True, verbose=False)


path_list = []
phase_out_path = SCANNET_OUT_PATH2
phase_out_path.mkdir(parents=True, exist_ok=True)
for f in (LABEL_PATH).glob('*'):
  scene = f.name
  path_list.append(scene + ',' + str(f / LABEL_NAME) + ',' + str(phase_out_path))

pool = ProcessPoolExecutor(max_workers=20)
result = list(pool.map(handle_process, path_list))
# for path in path_list:
#   handle_process(path)
