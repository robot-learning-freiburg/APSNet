# USAGE 

### Training
Train with a single GPU:
```
python tools/train.py ${CONFIG} --work_dir path/to/save/checkpoint/folder 
```
Train with multiple GPUS:
```
./tools/dist_train.sh ${CONFIG} ${GPU_NUM} --work_dir path/to/save/checkpoint/folder  
```
* `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
* `--validate`: Enables evaluation during the training process.

### Evaluation
Test with a single GPU:
```
python tools/test.py ${CONFIG} ${CHECKPOINT_FILE} --eval ${TASK}
```
Test with multiple GPUS:
```
./tools/dist_test.sh ${CONFIG} ${CHECKPOINT_FILE} ${GPU_NUM} --eval ${TASK}
```
- `${TASK}` can be either `panoptic` or `amodal-panoptic`.

### Inference
Saves predictions in expected dataset benchmarking structure. Set test_data_root, test dict img_prefix accordingly. panoptic_gt is ignored in this mode. 

Inference with a single GPU:
```
python tools/inference.py ${CONFIG} ${CHECKPOINT_FILE} --task ${TASK}
```
Inference with multiple GPUS:
```
./tools/dist_inference.sh ${CONFIG} ${CHECKPOINT_FILE} ${GPU_NUM} --task ${TASK}
```
- `${TASK}` can be either `panoptic` or `amodal-panoptic`.

### Inference
The inference process saves predictions in the structure expected by the dataset benchmark. Adjust `test_data_root` and the `img_prefix` in the test dictionary accordingly in the configuration file. Note that `panoptic_gt` is not used in this mode.

To perform inference with a single GPU:
```
python tools/inference.py ${CONFIG} ${CHECKPOINT_FILE} --task ${TASK}
```

To perform inference with multiple GPUs:
```
./tools/dist_inference.sh ${CONFIG} ${CHECKPOINT_FILE} ${GPU_NUM} --task ${TASK}
```

- `${TASK}` can be either `panoptic` or `amodal-panoptic`.


### Additional Notes:
#### Panoptic Segmentation
   * tools/visualization/cityscapes_save_predictions.py: saves color visualizations.
   * We only provide the single scale evaluation script. Multi-Scale+Flip evaluation further imporves the performance of the model.
   * This is a re-implementation of EfficientPS in PyTorch. The performance of the trained models might slightly differ from the metrics reported in the paper. Please refer to the metrics reported in [EfficientPS: Efficient Panoptic Segmentation](https://arxiv.org/abs/2004.02307) when making comparisons.