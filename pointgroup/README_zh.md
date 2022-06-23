# 在实例分割（Instance Segmentation）任务中使用PointGroup模型

[[English]](README.md)

本文件夹包含了在ScanNet数据集上使用虚假实例标签（Pseudo Instance Label）训练PointGroup模型的代码，我们从官方代码库[dvlab-research/PointGroup](https://github.com/dvlab-research/PointGroup)克隆了PointGroup代码，并在其基础上修改了数据加载模块，使得代码能够读取我们的虚假标签。

&nbsp;

## 安装

请根据[这里](https://github.com/dvlab-research/PointGroup#installation)的PointGroup安装步骤初始化你的环境，我们的代码使用和PointGroup一样的环境。

&nbsp;

## 数据处理

把程序运行目录改变至和ScanNet数据集相关的文件夹下。

```
cd dataset/scannetv2
```

使用如下命令产生使用真实标签（Ground-truth Label）的输入文件，你需要在`prepare_data_inst.py`文件中第27行和第29行指定数据集路径。

```
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```

使用如下命令产生使用虚假标签（Pseudo Label）的输入文件，你需要在`prepare_data_inst2.py`文件中第26行指定数据集路径，并在第31行指定虚假标签的路径。

```
python prepare_data_inst2.py --type manual
```

你可以根据你的需要改变虚假标签类别`--type`的赋值。

在数据处理完成后，将程序运行目录改变至主目录。

```
cd ../../
```

&nbsp;

## 开始训练

使用如下命令用真实标签去训练PointGroup模型。

```
python train.py --config config/pointgroup_run1_scannet.yaml 
```

使用如下命令用虚假标签去训练PointGroup模型，你需要确保`config/pointgroup_run2_scannet.yaml`文件中第11行的`type`的赋值和你在数据处理中`--type`的赋值是一致的。

```
python train.py --config config/pointgroup_run2_scannet.yaml 
```

训练过程仅使用了一个GPU卡，你可以通过在上面命令之前假如`CUDA_VISIBLE_DEVICES=0`指定使用GPU卡`0`，你可以根据你的需要改变使用的GPU卡的ID。

&nbsp;

## 评估

使用[这里](https://github.com/dvlab-research/PointGroup#inference-and-evaluation)的步骤去评估训练好的模型。对于使用虚假标签训练的模型，我们发现在第352个训练周期（Epoch）处，模型性能最佳。

我们训练好的模型可以在[清华云盘](https://cloud.tsinghua.edu.cn/f/e8a09b74ccbb4d3f81c6/)/[百度网盘](https://pan.baidu.com/s/1M1k9Yjw8IuysXIVDE1Rc0A)（提取密码：5q4o）下载，这个模型使用产生于手工标注的块级标签（Manually Labeled Seg-level Label）的虚假标签来训练，其性能如下：

| 划分 | AP | AP50 | AP25 |
| :---: | :---: | :---: | :---: | 
| 验证集 | 23.4 | 43.4 | 62.9 |
| 测试集 | 24.6 | 44.5 | 63.7 |
