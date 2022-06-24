# SegGroup

[[中文版]](README_zh.md)

<p float="left">
    <img src="image/SegGroup.png" width="800"/>
</p>

This repository contains the PyTorch implementation for paper **SegGroup: Seg-Level Supervision for 3D Instance and Semantic Segmentation**

**Authors:** An Tao, Yueqi Duan, Yi Wei, Jiwen Lu, Jie Zhou

[[arxiv]](https://arxiv.org/abs/2012.10217) [[FAQ]](FAQ.md)

If you find our work useful in your research, please consider citing:
```
@article{tao2020seggroup,
  title={{SegGroup}: Seg-Level Supervision for {3D} Instance and Semantic Segmentation},
  author={Tao, An and Duan, Yueqi and Wei, Yi and Lu, Jiwen and Zhou, Jie},
  journal={arXiv preprint arXiv:2012.10217},
  year={2020}
}
```

Our annotation tool is in [antao97/SegGroup.annotator](https://github.com/AnTao97/SegGroup.annotator).

**Updates:** 

- [2022/6/23] We update our paper in arXiv.

&nbsp;
## Usage

Our seg-level supervised point cloud segmentation method can be divided into two steps: 1) pseudo label generation with SegGroup and 2) fully-supervised point cloud segmentation model training with pseudo labels. The two stages are trained separately, and the evaluation of the segmentation performance is conducted on the model trained in step 2.

### 1. Pseudo Label Generation

Use our designed SegGroup model in [seggroup/](seggroup/) to generate point-level pseudo labels from seg-level labels.

### 2. Fully Supervised Point Cloud Segmentation Model Training

After generating pseudo labels, we can use them to replace the ground-truth labels on the training set to train a standard point cloud segmentation model with full supervision.

In our work, our pseudo labels can be used in both instance segmentation and semantic segmentation task.

#### Point Cloud Instance Segmentation

- [pointgroup/](pointgroup/)

#### Point Cloud Semantic Segmentation

- [kpconv/](kpconv/)
- [minkowski/](minkowski/)

&nbsp;
## Frequently Asked Questions

We show some some questions in [FAQ.md](FAQ.md).

- [Difference between SegGroup and OTOC?](https://github.com/antao97/SegGroup/blob/main/FAQ.md#difference-between-seggroup-and-otoc)
- [Details on over-segmentation?](https://github.com/antao97/SegGroup/blob/main/FAQ.md#details-on-over-segmentation)
- [How to determine the annotation rule?](https://github.com/antao97/SegGroup/blob/main/FAQ.md#how-to-determine-the-annotation-rule)
