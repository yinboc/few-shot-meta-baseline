## Meta-Dataset

The experiments on [Meta-Dataset](https://arxiv.org/abs/1903.03096).

The `meta_dataset` folder is cloned from [this repo](https://github.com/google-research/meta-dataset), its original data augmentation is disabled and the shuffle_buffer_size is set to be 300 in `data_config.gin`, drop_remainer for the batch loader is set to be True in `data/reader.py`.

### Results

Dataset|Classifier|Classifier(all)|Meta(all)
:-:|:-:|:-:|:-:
ILSVRC|59.2|55.0|48.0
Omniglot|69.1|76.9|89.4
Aircraft|54.1|69.8|81.7
Birds|77.3|78.3|77.3
Textures|76.0|71.4|64.5
Quick Draw|57.3|62.7|74.5
Fungi|45.4|55.4|60.2
VGG Flower|89.6|90.6|83.8
Traffic Signs|66.2|69.3|59.5
MSCOCO|55.7|53.1|43.6

### Running the code

Follow the instructions in [this repo](https://github.com/google-research/meta-dataset) and put the data in `./materials/records` with dataset folders in it.

Tensorflow 1.13 (CPU) is used for running data loader in `meta_dataset`.

Run `ulimit -n 100000` before running the code.

#### Training Classifier-Baseline on ILSVRC-2012
```
python train_classifier.py --config configs/train_classifier.yaml --gpu 0,1,2,3
```

#### Training Classifier-Baseline on All Datasets
```
python train_multi_classifier.py --config configs/train_multi_classifier.yaml --gpu 0,1,2,3
```

#### Training Meta-Baseline on All Datasets
```
python train_meta.py --config configs/train_meta.yaml --gpu 0
```

#### Test Model

Set `load_encoder`/`load` to the .pth file in `configs/test.yaml`, run
```
python train_meta.py --config configs/test.yaml --name _ --dataset {DATASET_NAME} --gpu 0
```
(Replace {DATASET_NAME} with one of: ilsvrc_2012, omniglot, aircraft, cu_birds, dtd, quickdraw, fungi, vgg_flower, traffic_sign, mscoco)
