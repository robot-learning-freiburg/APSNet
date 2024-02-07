# Dataset Preparation

## Training
- The dataset amodal annotations have to be converted into the COCO format using
`tools/datasets_converters/convert_{dataset_name}.py`:
```shell
python tools/convert_{dataset_name}.py path/to/root/folder/of/{dataset_name}/ ./data/{dataset_name}/
```
- The required directory structure is outlined below:
```shell
APSNet
├── mmdet
├── tools
├── configs
└── data
    └── dataset_name
        ├── annotations
        ├── train
        └── stuffthingmaps
```
Otherwise, in the configuration file, set `data_root` to the path of the `converted format` folder:
```shell
data_root = './data/{dataset_name}/'
```


## Validatation

### Amodal Panoptic Segmentation
-  The required directory structure for any dataset is outlined below:
```shell
DATASET
├── amodal_panoptic_seg
├── images
```
- In the configuration file, set `val_data_root` to dataset root folder.
```shell
val_data_root = 'path/to/root/folder/of/dataset/'
```
Otherwise, modify the data->val and test dictionaries by adjusting the img_prefix and panoptic_gt paths accordingly in the configuration file.

### Panoptic Segmentation
- Generate the panoptic labels for cityscapes dataset. 
```shell
python -m cityscapesscripts.preparation.createPanopticImgs --dataset-folder path/to/cityscapes/gtFine_folder --output-folder path/to/cityscapes/gtFine_folder --set-names val
```
- In the configuration file, set `val_data_root` to cityscapes root folder.
```shell
val_data_root = 'path/to/root/folder/of/cityscapes/'
```
Otherwise, modify the data->val and test dictionaries by adjusting the img_prefix and panoptic_gt paths accordingly in the configuration file.


