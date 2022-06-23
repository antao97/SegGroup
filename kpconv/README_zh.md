# 在语义分割（Semantic Segmentation）任务中使用KPConv模型

[[English]](README.md)

本文件夹包含了在ScanNet数据集上使用虚假语义标签（Pseudo Semantic Label）训练KPConv模型的代码，我们从官方代码库[HuguesTHOMAS/KPConv](https://github.com/HuguesTHOMAS/KPConv)克隆了KPConv代码，并在其基础上修改了数据加载模块，使得代码能够读取我们的虚假标签。

&nbsp;

## 准备工作

请根据[这里](https://github.com/HuguesTHOMAS/KPConv#installation)的KPConv安装步骤初始化你的环境，我们的代码使用和KPConv一样的环境。

你需要在`datasets/Scannet.py`文件的第142行和`datasets/Scannet2.py`文件的第299行给定ScanNet数据集的路径。

在`datasets/Scannet2.py`文件中我们使用了多重处理（Multiprocessing）以加快数据处理速度。

&nbsp;

## 开始训练

使用下面的命令用训练集上的真实标签训练KPConv模型，你需要在`training_Scannet.py`文件的第185行指定使用的显卡ID。

```
python training_Scannet.py
```

使用下面的命令用训练集上我们产生的虚假标签训练KPConv模型，你需要在`training_Scannet.py`文件的第185行指定使用的显卡ID，以及在第58行指定实验名称，该实验名称会被用于命名包含数据处理后的虚假标签的文件夹。

```
python training_Scannet2.py
```

因为在上面使用虚假标签的实验中需要使用验证集上的真实标签，所以你需要在运行上面的虚假标签实验之前首先运行使用真实标签的实验。

&nbsp;

## 评估

使用下面的命令在验证集上测试你训练好的模型性能，你需要在`test_any_model.py`文件的第53行指定使用的显卡ID，以及在第171行指定你需要评估的实验所在的路径。

```
python test_any_model.py
```

你可以在[清华云盘](https://cloud.tsinghua.edu.cn/f/0aff8985e7184e2b8d2f/)或者[百度网盘](https://pan.baidu.com/s/1KIZEEx4TVL4mDL79A6vCPw)（提取密码：0n3v）下载我们训练好的模型，该模型训练时使用了产生自手工标注的块级标签（Manual Labeled Seg-level Label）的虚假标签，模型性能如下：

| 划分 | mIoU |
| :---: | :---: | 
| 验证集 | 62.4 | 
| 测试集 | 61.1 | 
