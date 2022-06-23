# Use KPConv for Semantic Segmentation

[[中文版]](README_zh.md)

This folder contains codes to train KPConv on the ScanNet dataset with our pseudo semantic Labels. We clone the code from the official GitHub repo [HuguesTHOMAS/KPConv](https://github.com/HuguesTHOMAS/KPConv) and modify the data loading part to load our pseudo semantic labels. 

&nbsp;

## Preparation

Please follow the installation instructions of KPConv in [here](https://github.com/HuguesTHOMAS/KPConv#installation) to initialize your environment. We use the same environment as KPConv.

You need to specify your root path for the ScanNet dataset in line 142 of `datasets/Scannet.py` and line 299 of `datasets/Scannet2.py`.

In `datasets/Scannet2.py` we use multiprocessing to accerate the data preparation process.

&nbsp;

## Start Training

Use the following command to train KPConv with ground-truth labels on the training set. You need to specify the GPU card ID to use in line 185 of `training_Scannet.py`.

```
python training_Scannet.py
```

Use the following command to train KPConv with our generated pseudo labels on the training set. You need to specify the GPU card ID to use in Line 185 of `training_Scannet2.py`. Also, you need to specify an experiment name in line 58 of `training_Scannet2.py`. The specified experiment name is used to save processed pseudo labels for data loading.

```
python training_Scannet2.py
```

Because the above experiment with pseudo labels also uses ground-truth labels in the validation set, you need to first run the experiment with ground-truth labels before the above experiment with pseudo labels. 

&nbsp;

## Evaluation

Use the following command to evaluate the trained model on the validation set. You need to specify the GPU card ID to use in Line 53 of `test_any_model.py`. Also, you need to specify the path of the experiment to evaluate in line 171 of `test_any_model.py`.

```
python test_any_model.py
```

You can download our trained model in [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/0aff8985e7184e2b8d2f/) or [BaiduDisk](https://pan.baidu.com/s/1KIZEEx4TVL4mDL79A6vCPw) (Password: 0n3v). The model is trained with pseudo labels generated from manual labeled seg-level labels. Its performances are:

| Split | mIoU |
| :---: | :---: | 
| Validation Set | 62.4 | 
| Testing Set | 61.1 | 
