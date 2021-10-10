# Few-Shot Meta-Baseline

This repository contains the code for [Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning](https://arxiv.org/abs/2003.04390).

<img src="https://user-images.githubusercontent.com/10364424/76388735-bfb02580-63a4-11ea-8540-4021961a4fbe.png" width="600">

### Citation
```
@inproceedings{chen2021meta,
  title={Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning},
  author={Chen, Yinbo and Liu, Zhuang and Xu, Huijuan and Darrell, Trevor and Wang, Xiaolong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9062--9071},
  year={2021}
}
```

## Main Results

*The models on *miniImageNet* and *tieredImageNet* use ResNet-12 as backbone, the channels in each block are **64-128-256-512**, the backbone does **NOT** introduce any additional trick (e.g. DropBlock or wider channel in some recent work).*

#### 5-way accuracy (%) on *miniImageNet*

method|1-shot|5-shot
:-:|:-:|:-:
[Baseline++](https://arxiv.org/abs/1904.04232) |51.87|75.68|
[MetaOptNet](https://arxiv.org/abs/1904.03758) |62.64|78.63|
Classifier-Baseline |58.91|77.76|
Meta-Baseline |63.17|79.26|

#### 5-way accuracy (%) on *tieredImageNet*

method|1-shot|5-shot
:-:|:-:|:-:
[LEO](https://arxiv.org/abs/1807.05960) |66.33|81.44|
[MetaOptNet](https://arxiv.org/abs/1904.03758) |65.99|81.56|
Classifier-Baseline|68.07|83.74|
Meta-Baseline|68.62|83.29|

#### 5-way accuracy (%) on *ImageNet-800*

method|1-shot|5-shot
:-:|:-:|:-:
Classifier-Baseline (ResNet-18)|83.51|94.82|
Meta-Baseline (ResNet-18)|86.39|94.82|
Classifier-Baseline (ResNet-50)|86.07|96.14|
Meta-Baseline (ResNet-50)|89.70|96.14|

####

Experiments on Meta-Dataset are in [meta-dataset](https://github.com/cyvius96/few-shot-meta-baseline/tree/master/meta-dataset) folder.

## Running the code

### Preliminaries

**Environment**
- Python 3.7.3
- Pytorch 1.2.0
- tensorboardX

**Datasets**
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
- [ImageNet-800](http://image-net.org/challenges/LSVRC/2012/)

Download the datasets and link the folders into `materials/` with names `mini-imagenet`, `tiered-imagenet` and `imagenet`.
Note `imagenet` refers to ILSVRC-2012 1K dataset with two directories `train` and `val` with class folders.

When running python programs, use `--gpu` to specify the GPUs for running the code (e.g. `--gpu 0,1`).
For Classifier-Baseline, we train with 4 GPUs on miniImageNet and tieredImageNet and with 8 GPUs on ImageNet-800. Meta-Baseline uses half of the GPUs correspondingly.

In following we take miniImageNet as an example. For other datasets, replace `mini` with `tiered` or `im800`.
By default it is 1-shot, modify `shot` in config file for other shots. Models are saved in `save/`.

### 1. Training Classifier-Baseline
```
python train_classifier.py --config configs/train_classifier_mini.yaml
```

(The pretrained Classifier-Baselines can be downloaded [here](https://www.dropbox.com/sh/ef2sm8d8qadhg3a/AAAIBotzaCKIdN1dJTvgDk-wa?dl=0))

### 2. Training Meta-Baseline
```
python train_meta.py --config configs/train_meta_mini.yaml
```

### 3. Test
To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run
```
python test_few_shot.py --shot 1
```

## Advanced instructions

### Configs

A dataset/model is constructed by its name and args in a config file.

For a dataset, if `root_path` is not specified, it is `materials/{DATASET_NAME}` by default.

For a model, to load it from a specific saving file, change `load_encoder` or `load` to the corresponding path.
`load_encoder` refers to only loading its `.encoder` part.

In configs for `train_classifier.py`, `fs_dataset` refers to the dataset for evaluating few-shot performance.

In configs for `train_meta.py`, both `tval_dataset` and `val_dataset` are validation datasets, while `max-va.pth` refers to the one with best performance in `val_dataset`.

### Single-class AUC

To evaluate the single-class AUC, add `--sauc` when running `test_few_shot.py`.
