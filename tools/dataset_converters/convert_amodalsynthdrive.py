import argparse
from copy import copy
import json
from multiprocessing import Pool, Value, Lock
import os
import cv2
import pycocotools.mask as mask_utils


import numpy as np
import tqdm
from PIL import Image
from mmdet.core.evaluation.dataset_labels import asd_labels
from pycococreatortools import pycococreatortools as pct
from data_struct import read_asd_files


parser = argparse.ArgumentParser(description="Convert AmodalSynthDrive to coco format")
parser.add_argument(
    "root_dir", metavar="ROOT_DIR", type=str, help="Root directory of Cityscapes"
)
parser.add_argument("out_dir", metavar="OUT_DIR", type=str, help="Output directory")

# Add argument for image extension
parser.add_argument(
    "--image_ext",
    type=str,
    default="_rgb.jpg",
    help="Extension of the image files (default: _rgb.jpg)",
)

# Add argument for instance extension
parser.add_argument(
    "--instance_ext",
    type=str,
    default="_ampano.png",
    help="Extension of the instance files (default: _ampano.png)",
)

# Add argument for split JSON path
parser.add_argument(
    "--split_json_path",
    type=str,
    default=None,
    help="Path to the split JSON file (default: None)",
)


def main(args):
    """
    Convert AmodalSynthDrive dataset to COCO detection format.

    Args:
        args (object): Command-line arguments.

    Returns:
        None
    """
    print("Loading AmodalSynthDrive from", args.root_dir)
    ann_dir, stuff_dir = create_output_dir(args.out_dir)
    coco_categories = get_coco_categories()
    struct = read_asd_files(
        args.root_dir, args.image_ext, args.instance_ext, args.split_json_path
    )

    for split, split_info in struct.items():
        img_list = split_info["images"]
        ann_list = split_info["amodal_panoptic_seg"]
        img_ann_list = list(zip(img_list, ann_list))
        img_split_dir = os.path.join(args.out_dir, split)
        stuff_split_dir = os.path.join(stuff_dir, split)
        os.makedirs(img_split_dir, exist_ok=True)
        os.makedirs(stuff_split_dir, exist_ok=True)

        # Convert to COCO detection format
        coco_out = {
            "info": {"version": "1.0"},
            "images": [],
            "categories": coco_categories,
            "annotations": [],
        }
        # Process images in parallel
        worker = _Worker(img_split_dir, stuff_split_dir)
        with Pool(
            initializer=_init_counter, initargs=(_Counter(0),), processes=8
        ) as pool:
            total = len(img_ann_list)
            for coco_img, coco_ann in tqdm.tqdm(
                pool.imap(worker, img_ann_list, 8), total=total
            ):
                # COCO annotation
                coco_out["images"].append(coco_img)
                coco_out["annotations"] += coco_ann

        # Write COCO detection format annotation
        with open(os.path.join(ann_dir, split + ".json"), "w") as fid:
            json.dump(coco_out, fid)


def get_coco_categories():
    """
    Get the COCO categories from the ASD labels.

    Returns:
        coco_categories (list): List of dictionaries containing the thing category IDs and names.
    """
    coco_categories = []
    added_cls = []
    for _, lbl in asd_labels.items():
        if (
            lbl.trainId != 255
            and lbl.trainId != -1
            and lbl.hasInstances
            and lbl.trainId not in added_cls
        ):
            coco_categories.append({"id": lbl.trainId, "name": lbl.name})
            added_cls.append(lbl.trainId)
    return coco_categories


def create_output_dir(out_dir: str) -> tuple:
    """
    Create the output directory structure for annotations and stuffthingmaps.

    Args:
        out_dir (str): The path to the output directory.

    Returns:
        tuple: A tuple containing the paths to the annotations directory and the stuffthingmaps directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    stuff_dir = os.path.join(out_dir, "stuffthingmaps")
    os.makedirs(stuff_dir, exist_ok=True)
    return ann_dir, stuff_dir


class _Worker:
    """
    A worker class responsible for processing image and label files and generating COCO format annotations.

    Args:
        img_dir (str): The directory path of the output image files.
        msk_dir (str): The directory path to save the output mask files.

    Returns:
        tuple: A tuple containing the COCO format image annotation and a list of COCO format annotation dictionaries.
    """

    def __init__(self, img_dir: str, msk_dir: str):
        self.img_dir = img_dir
        self.msk_dir = msk_dir

    def __call__(self, img_desc: tuple):
        """
        Process the image and label files and generate COCO format annotations.

        Args:
            img_desc (tuple): A tuple containing the file paths of the image and label files.

        Returns:
            tuple: A tuple containing the COCO format image annotation and a list of COCO format annotation dictionaries.
        """
        img_file_path, lbl_file_path = img_desc
        new_img_file_name = ("_").join(img_file_path.split("/")[-2:])
        img_unique_id = new_img_file_name.split(".")[0]
        coco_ann = []
        # Load the annotation
        with Image.open(lbl_file_path) as lbl_img:
            lbl = np.array(lbl_img)
            lbl_size = lbl_img.size
        amodal_info = json.load(open(lbl_file_path.replace(".png", ".json")))
        ids = np.unique(lbl)
        # Compress the labels and compute cat
        lbl_out = np.ones(lbl.shape, np.uint8) * 255
        k = 0
        for city_id in ids:
            if city_id < 1000:
                # Stuff or group
                cls_i = city_id
                iscrowd_i = asd_labels[cls_i].hasInstances
            else:
                # Instance
                cls_i = city_id // 1000
                iscrowd_i = False

            # If it's a void class just skip it
            if asd_labels[cls_i].trainId == 255 or asd_labels[cls_i].trainId == -1:
                continue
            # Extract all necessary information
            iss_class_id = asd_labels[cls_i].trainId
            mask_i = lbl == city_id
            lbl_out[mask_i] = iss_class_id
            
            #Compute COCO detection format annotation
            if asd_labels[cls_i].hasInstances and city_id > 1000:
                category_info = {"id": iss_class_id, "is_crowd": iscrowd_i}
                coco_ann_i = pct.create_annotation_info(
                    counter.increment(),
                    img_unique_id,
                    category_info,
                    mask_i,
                    lbl_size,
                    tolerance=2,
                )
                if coco_ann_i is not None:
                    occ_flag = amodal_info[str(city_id)]["occluded"]
                    coco_ann_i["occluded"] = occ_flag
                    coco_ann_i["amodal_segmentation"] = copy(coco_ann_i["segmentation"])
                    coco_ann_i["amodal_bbox"] = copy(coco_ann_i["bbox"])
                    coco_ann_i["occluder_segmentation"] = []
                    coco_ann_i["occlusion_segmentation"] = []
                    if occ_flag:
                        amodal_mask = mask_utils.decode(
                            amodal_info[str(city_id)]["amodal_mask"]
                        ).astype(bool)
                        amodal_mask[lbl == 7] = False
                        if (np.sum(amodal_mask)) == 0:
                            continue
                        occ_mask = mask_utils.decode(
                            amodal_info[str(city_id)]["occlusion_mask"]
                        ).astype(bool)
                        occ_mask[lbl == 7] = False
                        if (np.sum(occ_mask)) == 0:
                            continue
                        coco_ann_a = pct.create_annotation_info(
                            10000,
                            img_unique_id,
                            category_info,
                            amodal_mask,
                            lbl_size,
                            tolerance=2,
                        )
                        coco_ann_i["amodal_bbox"] = copy(coco_ann_a["bbox"])
                        coco_ann_o = pct.create_annotation_info(
                            10000,
                            img_unique_id,
                            category_info,
                            occ_mask,
                            lbl_size,
                            tolerance=2,
                        )
                        if coco_ann_o is None or coco_ann_a is None:
                            continue
                        coco_ann_i["occlusion_segmentation"] = copy(
                            coco_ann_o["segmentation"]
                        )
                        coco_ann_i["amodal_segmentation"] = copy(
                            coco_ann_a["segmentation"]
                        )
                        occluder_mask = np.zeros(lbl.shape, np.uint8)
                        ref_mask = np.zeros(lbl.shape, np.uint8)
                        am_bbox = coco_ann_a["bbox"]
                        ref_mask[
                            int(am_bbox[1]) : int(am_bbox[1] + am_bbox[3]),
                            int(am_bbox[0]) : int(am_bbox[0] + am_bbox[2]),
                        ] = 1
                        for id_i in ids:
                            if id_i == city_id or id_i == 7:
                                continue
                            potential_occluder_mask = lbl == id_i
                            occluder_flag = np.logical_and(
                                potential_occluder_mask, amodal_mask
                            )
                            if np.sum(occluder_flag) == 0:
                                continue
                            occluder_mask_i = np.logical_and(
                                potential_occluder_mask, ref_mask
                            )
                            occluder_mask[occluder_mask_i] = 1
                        coco_ann_ocl = pct.create_annotation_info(
                            10000,
                            img_unique_id,
                            category_info,
                            occluder_mask.astype(bool),
                            lbl_size,
                            tolerance=2,
                        )
                        coco_ann_i["occluder_segmentation"] = copy(
                            coco_ann_ocl["segmentation"]
                        )
                    else:
                        coco_ann_i["occluded"] = 0
                    coco_ann.append(coco_ann_i)

        # COCO detection format image annotation
        coco_img = pct.create_image_info(img_unique_id, new_img_file_name, lbl_size)

        # Write output
        Image.fromarray(lbl_out).save(
            os.path.join(self.msk_dir, img_unique_id + ".png")
        )
        # Copy image
        temp_img = cv2.imread(img_file_path)
        cv2.imwrite(os.path.join(self.img_dir, new_img_file_name), temp_img)
        return coco_img, coco_ann


def _init_counter(c: int):
    """
    Initialize the counter variable.

    Args:
        c (int): The value to initialize the counter with.
    """
    global counter
    counter = c


class _Counter:
    """
    A class that represents a counter with thread-safe increment operation.

    Attributes:
        val (Value): A shared value representing the current counter value.
        lock (Lock): A lock object used for thread synchronization.
    """

    def __init__(self, initval: int = 0):
        self.val = Value("i", initval)
        self.lock = Lock()

    def increment(self):
        """
        Increments the counter value by 1 in a thread-safe manner.

        Returns:
            int: The previous value of the counter before incrementing.
        """
        with self.lock:
            val = self.val.value
            self.val.value += 1
        return val


if __name__ == "__main__":
    main(parser.parse_args())
