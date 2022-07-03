# Use Minkowski for Semantic Segmentation

[[中文版]](README_zh.md)

This folder contains codes to train Minkowski on the ScanNet dataset with our pseudo semantic Labels. We clone the code from the official GitHub repo [chrischoy/SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation) and modify the data loading part to load our pseudo semantic labels. 

&nbsp;

## Preparation

Please follow the installation instructions of Minkowski in [here](https://github.com/chrischoy/SpatioTemporalSegmentation#installation) to initialize your environment. We use the same environment as Minkowski. 

We recommend using [Anaconda](https://www.anaconda.com/) to install Minkowski Engine. Detailed steps can be found in [Minkowski Engine 0.4.3](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.4.3#anaconda). You need to make sure that you download the 0.4.3 version, not the latest version.

Here we provide an installation example in this [link](INSTALL.md).

## Dataset

Use the following command to preprocess all raw point clouds with ground-truth labels. You need to set the path `SCANNET_RAW_PATH` and `SCANNET_OUT_PATH` in lines 9 and 10 of `lib/datasets/preprocessing/scannet.py`.

```
python -m lib.datasets.preprocessing.scannet
```

Use the following command to preprocess all raw point clouds with pseudo labels. You need to set `EXP_NAME`, `LABEL_NAME`, `SCANNET_RAW_PATH2`, `LABEL_PATH`, and `SCANNET_OUT_PATH2` in lines 7-12 of `lib/datasets/preprocessing/scannet2.py`. The experiment name `EXP_NAME` of pseudo labels corresponds to label style `<label style>` in SegGroup.

```
cp -r SCANNET_OUT_PATH/train SCANNET_OUT_PATH2
python -m lib.datasets.preprocessing.scannet2
```

&nbsp;

## Start Training

Use the following command to train Minkowski with ground-truth labels on the training set. 

```
export BATCH_SIZE=8;
./scripts/train_scannet.sh <gpu id> \
        -default \
        "--scannet_path SCANNET_OUT_PATH/train"
```

Use the following command to train Minkowski with pseudo labels on the training set. 

```
export BATCH_SIZE=8;
./scripts/train_scannet.sh <gpu id> \
        -default \
        "--scannet_path SCANNET_OUT_PATH2"
```

You can only use one GPU ID `<gpu id>`. We use one 32G Tesla V100 GPU in our experiment.

&nbsp;

## Evaluation

Use the following command to evaluate the trained model on the validation set. 

```
python test_scannet.py --weights <weight path> --data_path SCANNET_OUT_PATH/train
```

The format of `<weight path>` is `outputs/ScannetVoxelization2cmDataset/Res16UNet34C-b8-120000--default/<exp name>/checkpoint_NoneRes16UNet34Cbest_val.pth`.

You can download our trained model in [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/97b1f26fcc8140b3802a/) or [BaiduDisk](https://pan.baidu.com/s/18yCP5-hvheg0YfZBZ8lU_g) (Password: wwm2). The model is trained with pseudo labels generated from manual labeled seg-level labels. Its performances are:

| Split | mIoU |
| :---: | :---: | 
| Validation Set | 64.5 | 
| Testing Set | 62.7 | 
