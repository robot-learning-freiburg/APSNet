import json
import os

import numpy as np
import torch.distributed as dist
from PIL import Image


def create_dir(eval_dir: str) -> tuple:
    base_path = os.path.join(eval_dir, 'tmp')
    base_json = os.path.join(eval_dir, 'tmp_json')
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(base_json, exist_ok=True)

    return base_path, base_json


def save_panoptic_eval(results: list, eval_dir: str, label_info: dict) -> None:
    base_path, base_json = create_dir(eval_dir)
    for result in results:
        images = []
        annotations = []
        pan_pred, cat_pred, meta = result
        pan_pred, cat_pred = pan_pred.numpy(), cat_pred.numpy()
        pan_format = np.zeros(
            (pan_pred.shape[0], pan_pred.shape[1], 3), dtype=np.uint8
        )

        imgName = meta[0]['filename'].split('/')[-1]
        imageId = meta[0]['image_id']
        inputFileName = imgName
        outputFileName = '{}_panoptic.png'.format(imageId)
        images.append({"id": imageId,
                       "width": int(pan_pred.shape[1]),
                       "height": int(pan_pred.shape[0]),
                       "file_name": inputFileName})
        segmInfo = []
        for panPredId in np.unique(pan_pred):
            if cat_pred[panPredId] == 255:
                continue

            semanticId = label_info[cat_pred[panPredId]].id

            segmentId = semanticId * 1000 + panPredId

            isCrowd = 0
            categoryId = semanticId

            mask = pan_pred == panPredId
            color = [segmentId % 256, segmentId //
                     256, segmentId // 256 // 256]
            pan_format[mask] = color

            area = np.sum(mask)

            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]

            segmInfo.append({"id": int(segmentId),
                             "category_id": int(categoryId),
                             "area": int(area),
                             "bbox": bbox,
                             "iscrowd": isCrowd})
        annotations.append({'image_id': imageId,
                            'file_name': outputFileName,
                            "segments_info": segmInfo})

        Image.fromarray(pan_format).save(
            os.path.join(base_path, outputFileName))
        d = {'images': images,
             'annotations': annotations,
             'categories': {}}
        with open(os.path.join(base_json, imageId + '.json'), 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)
