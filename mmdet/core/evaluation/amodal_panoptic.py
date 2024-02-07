import json
import os

import cv2
import numpy as np
import torch.distributed as dist
from pycocotools import mask as mask_utils


def create_dir(eval_dir: str) -> str:
    """Create directory containing to save amodal panoptic evaluation results."""
    tmp_dir_name = os.path.join(eval_dir, 'amodal_panoptic_seg')
    os.makedirs(tmp_dir_name, exist_ok=True)
    return tmp_dir_name

def encode_amodal_mask(amodal_mask: np.ndarray) -> dict:
    amodal_mask = mask_utils.encode(np.asfortranarray(amodal_mask))
    amodal_mask['counts'] = amodal_mask['counts'].decode('utf-8')
    return amodal_mask


def save_amodal_panoptic_eval(results: list, eval_dir: str, label_info: dict) -> None:
    base_path = create_dir(eval_dir)

    for result in results:
        pan_pred, cat_pred, amodal_pred, meta = result
        pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()
        if amodal_pred is not None:
            amodal_pred = amodal_pred.numpy().astype(np.uint8)
        pan_format = np.zeros(
            (pan_pred.shape[0], pan_pred.shape[1]), dtype=np.uint16
        )
        parts = meta[0]['filename'].split('/')
        position = parts.index('images')
        seq_path = base_path
        for i in range(position + 1, len(parts) - 1):
            seq_path = os.path.join(seq_path, parts[i])
        os.makedirs(seq_path, exist_ok=True)
        imageId = meta[0]['image_id']
        outputFileName = '{}_ampano.png'.format(imageId)
        segmInfo = {}
        for panPredId in np.unique(pan_pred):
            if cat_pred[panPredId] == 255:
                continue
            
            semanticId = label_info[cat_pred[panPredId]].id
            mask = pan_pred == panPredId
            if not label_info[cat_pred[panPredId]].hasInstances:
                pan_format[mask] = semanticId
                continue
            
            segmentId = semanticId * 1000 + panPredId
            pan_format[mask] = segmentId

            amodal_mask = amodal_pred[panPredId]
            amodal_mask = encode_amodal_mask(amodal_mask)
            segmInfo[int(segmentId)] = {"category_id": int(semanticId),
            "iscrowd": 0,
            "occlusion_mask": [],
            "amodal_mask": amodal_mask,
            }

        cv2.imwrite(os.path.join(seq_path, outputFileName), pan_format)
        with open(os.path.join(seq_path, imageId + '_ampano.json'), 'w') as f:
            json.dump(segmInfo, f)
