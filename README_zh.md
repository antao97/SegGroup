# SegGroup

[[English]](README.md)

<p float="left">
    <img src="image/SegGroup.png" width="800"/>
</p>

本代码库包含了文章**SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation**的PyTorch实现代码

**作者：** 陶安，段岳圻，韦祎，鲁继文，周杰

[[论文]](https://arxiv.org/abs/2012.10217) [[知乎专栏]](https://zhuanlan.zhihu.com/p/536482202) [[常见问题]](FAQ.md)

如果您发现我们的工作对您的研究有帮助，您可以考虑引用我们的论文。
```
@article{tao2022seggroup,
  title={{SegGroup}: Seg-Level Supervision for {3D} Instance and Semantic Segmentation},
  author={Tao, An and Duan, Yueqi and Wei, Yi and Lu, Jiwen and Zhou, Jie},
  journal={IEEE Transactions on Image Processing},
  year={2022},
  volume={31},
  pages={4952-4965},
  publisher={IEEE}
}
```

我们的标注工具在[antao97/SegGroup.annotator](https://github.com/AnTao97/SegGroup.annotator)。

**更新：** 

- [2022/07/01] 这份工作被IEEE Transactions on Image Processing接收了！
- [2022/06/23] 我们在arXiv更新了文章。

&nbsp;
## 使用说明

我们块级监督（Seg-level Supervised）的点云分割方法可以分为两部分：1）使用SegGroup模型产生虚假标签，2）使用虚假标签（Pseudo Label）的全监督点云分割模型训练。这两部分是分别训练的，测试分割性能时使用步骤2中训练得到的模型。

### 1. 虚假标签生成

在[seggroup/](seggroup/)文件夹中，使用我们设计的SegGroup模型从块级标签（Seg-level Label）产生点级虚假标签（Point-level Pseudo Label）。

### 2. 全监督点云分割模型训练

产生虚假标签之后，我们可以用他们来替代训练集上的真实标签，从而训练一个标准的全监督点云分割模型

在我们的工作中，我们的虚假标签可以被用于实例分割和语义分割。

#### 点云实例分割

- [pointgroup/](pointgroup/)

#### 点云语义分割

- [kpconv/](kpconv/)
- [minkowski/](minkowski/)

&nbsp;
## 常见问题

- [SegGroup与OTOC的区别？](https://github.com/antao97/SegGroup/blob/main/FAQ.md#difference-between-seggroup-and-otoc)
- [过分割的细节？](https://github.com/antao97/SegGroup/blob/main/FAQ.md#details-on-over-segmentation)
- [如何确定标注规则？](https://github.com/antao97/SegGroup/blob/main/FAQ.md#how-to-determin-the-annotation-rule)
