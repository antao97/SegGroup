# Instance Segmentation with PointGroup

[[中文版]](README_zh.md)

This folder contains codes to train PointGroup on the ScanNet dataset with our pseudo semantic Labels. We clone the code from the official GitHub repo [dvlab-research/PointGroup](https://github.com/dvlab-research/PointGroup) and modify the data loading part to load our pseudo semantic labels. 

&nbsp;

## Installation

Please follow the installation instructions of PointGroup in [here](https://github.com/dvlab-research/PointGroup#installation) to initialize your environment. We use the same environment as PointGroup.

&nbsp;

## Data Preparation

Change your directory into the dataset folder.

```
cd dataset/scannetv2
```

Use the following command to generate input files with ground-truth labels. You need to change the dataset path in line 27 and 29 of `prepare_data_inst.py`.

```
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

Use the following command to generate input files with pseudo labels. You need to change the dataset path in line 26 of `prepare_data_inst2.py`. You also need to specify the pseudo label path in line 31 of `prepare_data_inst2.py`. 

```
python prepare_data_inst2.py --type manual
```

You can change the pseudo label type `--type` to match your own need.

After data preparation, change your directory back.

```
cd ../../
```

&nbsp;

## Start Training

Use the following command to use ground-truth labels on the training set to train PointGroup. 

```
python train.py --config config/pointgroup_run1_scannet.yaml 
```

Use the following command to use pseudo labels on the training set to train PointGroup. You need to make sure `type` in line 11 of `config/pointgroup_run2_scannet.yaml` the same as the type you use in `--type` for data preparation.

```
python train.py --config config/pointgroup_run2_scannet.yaml 
```

The training process only use one GPU card. To specify the card to use, add `CUDA_VISIBLE_DEVICES=0` before each command to use GPU card `0`. You can change the card ID depending on your own needs.

&nbsp;

## Evaluation

Follow the instructions in [here](https://github.com/dvlab-research/PointGroup#inference-and-evaluation) to evaluate the trained model. For the model trained with pseudo labels, we found the model at epoch 352 performs the best.

You can download our trained model in [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e8a09b74ccbb4d3f81c6/) or [BaiduDisk](https://pan.baidu.com/s/1M1k9Yjw8IuysXIVDE1Rc0A) (Password: 5q4o). This model is trained with pseudo labels generated from manually labeled seg-level labels. The performances are:

| Split | AP | AP50 | AP25 |
| :---: | :---: | :---: | :---: | 
| Validation Set | 23.4 | 43.4 | 62.9 |
| Testing Set | 24.6 | 44.5 | 63.7 |
