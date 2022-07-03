# 在语义分割（Semantic Segmentation）任务中使用Minkowski模型

[[English]](README.md)

本文件夹包含了在ScanNet数据集上使用虚假语义标签（Pseudo Semantic Label）训练Minkowski模型的代码，我们从官方代码库[chrischoy/SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation)克隆了Minkowski代码，并在其基础上修改了数据加载模块，使得代码能够读取我们的虚假标签。

&nbsp;

## 准备工作

请根据[这里](https://github.com/chrischoy/SpatioTemporalSegmentation#installation)的Minkowski安装步骤初始化你的环境，我们的代码使用和Minkowski一样的环境。

我们推荐您使用[Anaconda](https://www.anaconda.com/)来安装Minkowski Engine，相关的安装步骤可以参考[Minkowski Engine 0.4.3](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.4.3#anaconda)代码库。你需要确保你下载的是0.4.3版本安装包，不是最新版。

在[这里](INSTALL_zh.md)我们提供了一份安装示例。

## 数据集

使用下面的命令预处理所有原始点云及其真实标签（Ground-truth Label），你需要在`lib/datasets/preprocessing/scannet.py`文件的第9行和第10行对`SCANNET_RAW_PATH`和`SCANNET_OUT_PATH`设置好路径。

```
python -m lib.datasets.preprocessing.scannet
```

使用下面的命令预处理所有原始点云及其虚假标签（Pseudo Label），你需要在`lib/datasets/preprocessing/scannet2.py`文件的第7-12行对`EXP_NAME`、`LABEL_NAME`进行设置，对`SCANNET_RAW_PATH2`、`LABEL_PATH`和`SCANNET_OUT_PATH2`设置好路径。虚假标签的实验名称`EXP_NAME`对应于SegGroup中的标签风格`<label style>`。

```
cp -r SCANNET_OUT_PATH/train SCANNET_OUT_PATH2
python -m lib.datasets.preprocessing.scannet2
```

&nbsp;

## 开始训练

使用下面的命令用训练集上的真实标签训练Minkowski模型。

```
export BATCH_SIZE=8;
./scripts/train_scannet.sh <gpu id> \
        -default \
        "--scannet_path SCANNET_OUT_PATH/train"
```

使用下面的命令用训练集上的虚假标签训练Minkowski模型。

```
export BATCH_SIZE=8;
./scripts/train_scannet.sh <gpu id> \
        -default \
        "--scannet_path SCANNET_OUT_PATH2"
```

你只能使用一个显卡ID`<gpu id>`，在我们的实验中我们使用一张32G的Tesla V100显卡。

&nbsp;

## Evaluation

使用下面的命令在验证集上测试你训练好的模型性能。

```
python test_scannet.py --weights <weight path> --data_path SCANNET_OUT_PATH/train
```

模型路径`<weight path>`的格式是`outputs/ScannetVoxelization2cmDataset/Res16UNet34C-b8-120000--default/<exp name>/checkpoint_NoneRes16UNet34Cbest_val.pth`。

你可以在[清华云盘](https://cloud.tsinghua.edu.cn/f/97b1f26fcc8140b3802a/)或者[百度网盘](https://pan.baidu.com/s/18yCP5-hvheg0YfZBZ8lU_g)（提取密码：wwm2）下载我们训练好的模型，该模型训练时使用了产生自手工标注的块级标签（Manual Labeled Seg-level Label）的虚假标签，模型性能如下：

| 划分 | mIoU |
| :---: | :---: | 
| 验证集 | 64.5 | 
| 测试集 | 62.7 | 
