# Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification (ICME 2026)
This repository provides the official implementation of our paper:
> **[Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification](https://arxiv.org/pdf/2505.21854)**
> Jun Chen<sup>1*</sup>, Xinke Li<sup>2*</sup>, Mingyue Xu<sup>1</sup>, Chongshou Li<sup>1†</sup>, Tianrui Li<sup>1</sup>

## Requirements

* tqdm >= 4.52.0
* numpy >= 1.19.2
* scipy >= 1.6.3
* open3d >= 0.13.0
* torchvision >= 0.7.0
* scikit-learn >= 1.0

## Datasets and Models

Please download the dataset from [ModelNet40](https://aistudio.baidu.com/datasetdetail/35331).
Please download the [pretrained models](https://drive.google.com/file/d/1L25i0l6L_b1Vw504WQR8-Z0oh2FJA0G9/view?usp=sharing) and put them under ```/checkpoint```.
The victim models we use in the experiments are [PointNet](https://github.com/charlesq34/pointnet), [DGCNN](https://github.com/WangYueFt/dgcnn), 
[PCT](https://github.com/MenghaoGuo/PCT) and [CurveNet](https://curvenet.github.io/).

## Example Usage

### Generate adversarial examples by attacking PointNet:

```
python main.py --dataset ModelNet40 --data_path /your/path/to/dataset/ --transfer_attack_method WAAttack --surrogate_model pointnet_cls --target_model pointnet_cls
```
