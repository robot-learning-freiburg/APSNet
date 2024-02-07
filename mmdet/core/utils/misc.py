from functools import partial

import os
import mmcv
import cv2
import numpy as np
import json
from six.moves import map, zip
from skimage.segmentation import find_boundaries
import shutil


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def visualize_panoptic_prediction(
    img_path,
    panoptic,
    cat,
    trainId2label,
    overlay=True,
    overlay_opacity=0.3,
):
    """
    Visualizes the panoptic prediction on an image.

    Args:
        img_path (str): The path to the input image.
        panoptic (ndarray): The panoptic prediction array.
        cat (ndarray): The category array.
        trainId2label (dict): The mapping of train IDs to labels.
        overlay (bool, optional): Whether to overlay the prediction on the image. Defaults to True.
        overlay_opacity (float, optional): The opacity of the overlay. Defaults to 0.3.

    Returns:
        ndarray: The visualized panoptic prediction.
    """
    panoptic = panoptic
    cat = cat.numpy()
    img = mmcv.imread(img_path)
    visualize_pred = np.zeros((panoptic.shape[0], panoptic.shape[1], 3), dtype=np.uint8)
    visualize_boundaries = np.zeros(
        (panoptic.shape[0], panoptic.shape[1]), dtype=np.uint8
    )

    for i in np.unique(panoptic):
        if cat[i] == 255:
            continue
        id = cat[i]
        id_mask = panoptic == i
        color = np.array(trainId2label[id].color)[::-1]
        if trainId2label[id].hasInstances:
            contours = (
                find_boundaries(id_mask, mode="outer", background=0).astype(np.uint8)
                * 255
            )
            visualize_boundaries[contours != 0] = 255
        visualize_pred[id_mask] = color

    visualize_pred[visualize_boundaries != 0] = np.uint8(np.array([255, 255, 255]))

    if overlay:
        visualize_pred = cv2.addWeighted(
            img, overlay_opacity, visualize_pred, 1 - overlay_opacity, 0
        )

    return visualize_pred


def visualize_amodal_panoptic_prediction(
    img_path,
    panoptic,
    cat,
    amodal,
    trainId2label,
    amodal_opacity=0.5,
    overlay=True,
    overlay_opacity=0.3,
):
    """
    Visualizes the amodal panoptic prediction.

    Args:
        img_path (str): The path to the input image.
        panoptic (numpy.ndarray): The panoptic segmentation map.
        cat (numpy.ndarray): The category map.
        amodal (numpy.ndarray): The amodal mask.
        trainId2label (dict): The mapping of train IDs to labels.
        amodal_opacity (float, optional): The opacity of the amodal mask. Defaults to 0.5.
        overlay (bool, optional): Whether to overlay the visualization on the input image. Defaults to True.
        overlay_opacity (float, optional): The opacity of the overlay. Defaults to 0.3.

    Returns:
        numpy.ndarray: The visualized amodal panoptic prediction.
    """
    panoptic = panoptic
    amodal = amodal
    cat = cat
    img = mmcv.imread(img_path)
    visualize_pred = np.zeros((panoptic.shape[0], panoptic.shape[1], 3), dtype=np.uint8)

    for i in np.unique(panoptic):
        if cat[i] == 255:
            continue
        id = cat[i]
        id_mask = panoptic == i
        color = np.array(trainId2label[id].color)[::-1]
        if trainId2label[id].hasInstances:
            id_mask = (amodal[i].copy() != 0).astype(np.uint8)
            contours = (
                find_boundaries(id_mask, mode="outer", background=0).astype(np.uint8)
                * 255
            )
            id_mask[contours != 0] = 0
            visualize_pred[id_mask] = visualize_pred[
                id_mask
            ] * amodal_opacity + color * (1 - amodal_opacity)
            visualize_pred[contours != 0] = np.uint8(np.array([255, 255, 255]))
        else:
            visualize_pred[id_mask] = color

    if overlay:
        visualize_pred = cv2.addWeighted(
            img, overlay_opacity, visualize_pred, 1 - overlay_opacity, 0
        )

    return visualize_pred


def prepare_cityscapes_benchmarking(out, split):
    """
    Prepare the Cityscapes benchmarking data for evaluation.

    Args:
        out (str): The output directory path.
        split (str): The split name. Eg. val, test

    Returns:
        None
    """
    os.rename(
        os.path.join(out, "tmp"), os.path.join(out, f"cityscapes_panoptic_{split}")
    )

    pred_json = os.path.join(out, "tmp_json")
    pred_dict = {"images": [], "annotations": [], "categories": {}}
    for pred_ann in sorted(os.listdir(pred_json)):
        with open(os.path.join(pred_json, pred_ann), "r") as f:
            tmp_json = json.load(f)

        pred_dict["images"].extend(tmp_json["images"])
        pred_dict["annotations"].extend(tmp_json["annotations"])

    with open(os.path.join(out, f"cityscapes_panoptic_{split}.json"), "w") as f:
        json.dump(pred_dict, f)

    shutil.rmtree(pred_json)
