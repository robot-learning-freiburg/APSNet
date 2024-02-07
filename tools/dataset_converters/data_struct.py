# AmodalSynthDriveDataset folder structure:
# amodal_synth_drive
# ├── train
# │   ├── images
# │         ├── front
# │         ├── back
# │         ├── left
# │         ├── right
# │   ├── amodal_panoptic_seg
# │         ├── front
# │         ├── back
# │         ├── left
# │         ├── right
# ├── val
# │   ├── images
#           ├── ...
# │   ├── amodal_panoptic_seg
#           ├── ...
# ├── test
# │   ├── images
#           ├── ...
# │   ├── amodal_panoptic_seg
#           ├── ...
import os
import glob
import json
from typing import Optional


def read_asd_files(
    root_dir: str,
    img_ext: str = "_rgb.jpg",
    lbl_ext: str = "_ampano.png",
    splits_path: Optional[str] = None,
):
    """
    Reads ASD files from the specified root directory.

    Args:
        root_dir (str): The root directory containing the ASD files.
        img_ext (str, optional): The image file extension. Defaults to "_rgb.jpg".
        lbl_ext (str, optional): The label file extension. Defaults to "_ampano.png".
        splits_path (str, optional): The path to the splits file. Defaults to None.

    Returns:
        dict: A dictionary containing the ASD file structure.
    """
    if splits_path is None:
        struct = {"train": {}, "val": {}, "test": {}}
        splits = list(struct.keys())
        for split in splits:
            split_gt_dir = os.path.join(root_dir, split, "amodal_panoptic_seg")
            if not os.path.exists(split_gt_dir):
                struct.pop(split)
                continue
            amodal_pan_files = sorted(
                glob.glob(os.path.join(split_gt_dir, "*", "*", "*" + lbl_ext))
            )
            struct[split]["amodal_panoptic_seg"] = amodal_pan_files
            struct[split]["images"] = [
                f.replace("amodal_panoptic_seg", "images").replace(lbl_ext, img_ext)
                for f in amodal_pan_files
            ]
    else:
        splits = json.load(open(splits_path))
        struct = {split: {} for split in splits}
        for split in splits:
            split_gt_dir = os.path.join(root_dir, "amodal_panoptic_seg")
            sequences = splits[split]
            amodal_pan_files = []
            for sequence in sequences:
                amodal_pan_files += glob.glob(
                    os.path.join(split_gt_dir, "*", sequence, "*" + lbl_ext)
                )
            amodal_pan_files = sorted(amodal_pan_files)
            struct[split]["amodal_panoptic_seg"] = amodal_pan_files
            struct[split]["images"] = [
                f.replace("amodal_panoptic_seg", "images").replace(lbl_ext, img_ext)
                for f in amodal_pan_files
            ]
    return struct
