# APSNet: Amodal Panoptic Segmentation

APSNet is a top-down approach for amodal panoptic segmentation, where the goal is to concurrently predict the pixel-wise semantic segmentation labels of visible regions of "stuff" classes (e.g., road, sky, and so on), and instance segmentation labels of both the visible and occluded regions of "thing" classes (e.g., car, truck, etc).

![Illustration of Amodal Panoptic Segmentation task](/images/intro.png)

This repository contains the **PyTorch implementation** of our CVPR'2022 paper [Amodal Panoptic Segmentation](https://arxiv.org/pdf/2202.11542.pdf). The repository builds on [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS), [mmdetection](https://github.com/open-mmlab/mmdetection), [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch), and [pycls](https://github.com/facebookresearch/pycls) codebases.

Additionally, this repository includes an implementation of [EfficientPS](https://github.com/DeepSceneSeg/EfficientPS) compatible with PyTorch version 1.12.

If you find this code useful for your research, we kindly ask you to consider citing our papers:

For Amodal Panoptic Segmentation:
```
@article{mohan2022amodal,
  title={Amodal panoptic segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21023--21032},
  year={2022}
}
```
For EfficientPS: Efficient Panoptic Segmentation:
```
@article{mohan2020efficientps,
  title={Efficientps: Efficient panoptic segmentation},
  author={Mohan, Rohit and Valada, Abhinav},
  journal={International Journal of Computer Vision (IJCV)},
  year={2021}
}
```

## System Requirements
* Linux 
* Python 3.9
* PyTorch 1.12.1
* CUDA 11
* GCC 7 or 8

**IMPORTANT NOTE**: These requirements are not necessarily mandatory. However, we have only tested the code under the above settings and cannot provide support for other setups.

##  Installation
Please refer to the [installation documentation](https://github.com/robot-learning-freiburg/APSNet/blob/main/docs/INSTALLATION.md) for detailed instructions.

## Dataset Preparation
Please refer to the [dataset documentation](https://github.com/robot-learning-freiburg/APSNet/blob/main/docs/DATASET.md) for detailed instructions.

## Usage
For detailed instructions on training, evaluation, and inference processes, please refer to the [usage documentation](https://github.com/robot-learning-freiburg/APSNet/blob/main/docs/USAGE.md).


## Pre-Trained Models
Pre-trained models can be found in the [model zoo](https://github.com/robot-learning-freiburg/APSNet/blob/main/docs/MODELS.md).

## Acknowledgements
We have used utility functions from other open-source projects. We espeicially thank the authors of:
- [mmdetection](https://github.com/open-mmlab/mmdetection)
- [gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch)
- [pycls](https://github.com/facebookresearch/pycls)

## Contacts
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
* [Rohit Mohan](https://github.com/mohan1914)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.

