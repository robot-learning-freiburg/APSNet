#  Installation
a. Create a conda virtual environment and activate it.
```shell
conda create --name apsnet python=3.9
conda activate apsnet
```
b. Install torch
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
c. Install dependencies:
```shell
pip install -U openmim
mim install mmcv-full==1.7.0

git clone https://github.com/waspinator/pycococreator.git
cd pycococreator
pip install .
cd ..
```
d. Download the repository
```shell
git clone https://github.com/DeepSceneSeg/APSNet.git
cd APSNet
```
e. Install all other dependencies using pip:
```bash
pip install -r requirements.txt
```
f. Install EfficientNet implementation
```bash
cd efficientNet
pip install .
cd ..
```
g. Install Pycls
```bash
cd pycls
pip install .
cd ..
```
h. If using cityscapes dataset:
```shell
pip install cityscapesscripts
```
i. Install
```shell
pip install -e .
```