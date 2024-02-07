import argparse
import os

import mmcv
import torch
import numpy as np
import json
import cv2

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
#from tools.fuse_conv_bn import fuse_module

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.cityscapes import PALETTE as PALETTE
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
from mmdet.core import cityscapes_originalIds

from PIL import Image
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries

map = {0:0,
       1:1,
       2:2,
       3:3,
       4:4,
       5:5,
       6:6,
       7:7,
       8:8,
       9:9,
       10:10,
       11:12,
       12:14,
       13:15,
       14:16,
       15:17,
       16:18,
       17:11,
       18:13,
       255:255,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input', help='input folder')
    parser.add_argument('out', help='output folder')
    parser.add_argument('--color', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    model = init_detector(args.config, args.checkpoint, device='cuda:0')

    images = []
    annotations = []
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    PALETTE.append([0,0,0])
    colors = np.array(PALETTE, dtype=np.uint8)

    for city in os.listdir(args.input):
        if city != 'vis_raw':
            continue
        path = os.path.join(args.input, city)
        out_dir = os.path.join(args.out, city)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        prog_bar = mmcv.ProgressBar(len(os.listdir(path)))
        for imgName in os.listdir(path):
            result = inference_detector(model, os.path.join(path, imgName), eval='panoptic')
            pan_pred, cat_pred, _, sem_pred_, depth_ = result[0]
            imageId = imgName.replace(".png", "")
            inputFileName = imgName
            outputFileName = imgName.replace(".png", ".png")
            img = Image.fromarray(cv2.imread(os.path.join(path, imgName))[:, :, ::-1])
#            cv2.imshow('img', np.uint8(np.array(img)))
#            cv2.waitKey(0)

            depth_ = depth_.sigmoid() #[0,0,...].cpu().numpy()
            depth_ = np.uint16(np.round(88*256*depth_[0,0,...].cpu().numpy()))
            depth1_ = cv2.resize(depth_, img.size, interpolation = cv2.INTER_NEAREST)

            np.save(os.path.join('raw_depth', outputFileName.replace('.png','.npy')), depth1_) 


            depth_ = np.uint8(255*(depth_/float(np.max(depth_))))
            depth_ = cv2.resize(depth_, img.size, interpolation = cv2.INTER_NEAREST)
            depth_img = cv2.applyColorMap(depth_, cv2.COLORMAP_JET)
            #cv2.imwrite(os.path.join('color_depth', outputFileName), depth_img)
            #cv2.imwrite(os.path.join('gray_depth', outputFileName), depth_)
            continue
            print(pan_pred.shape, pan_pred.numpy().dtype)
#            pan_pred_ = cv2.resize(pan_pred.numpy(), img.size, interpolation = cv2.INTER_NEAREST)
#            sem = cat_pred[pan_pred].numpy()
#            sem_m = sem_pred_.numpy()
#            sem[sem==255] = sem_m[sem==255]
#            sem_ = cv2.resize(sem, img.size, interpolation = cv2.INTER_NEAREST)

            if args.color:
                img = Image.open(os.path.join(path, imgName))
                out_path = os.path.join(out_dir, outputFileName)
                print ('cat_pred', cat_pred)
                sem = cat_pred[pan_pred].numpy()
                sem_m = sem_pred_.numpy()
                sem[sem==255] = sem_m[sem==255] # use raw segmentation prediction
                sem = cv2.resize(sem, img.size, interpolation = cv2.INTER_NEAREST)
                print ('np.unique(sem)', np.unique(sem))
                sem_tmp = sem.copy()
                sem_tmp[sem==255] = colors.shape[0] - 1
                sem_img = Image.fromarray(colors[sem_tmp])
                cv2.imshow('sem_img', np.uint8(np.array(sem_img)))
                cv2.waitKey(0)

                is_background = (sem < 11) | (sem == 255)
                pan_pred = pan_pred.numpy()
                pan_pred = cv2.resize(pan_pred, img.size, interpolation = cv2.INTER_NEAREST) 
                pan_pred[is_background] = 0
                contours = find_boundaries(pan_pred, mode="outer", background=0).astype(np.uint8) * 255
                contours = dilation(contours)

                contours = np.expand_dims(contours, -1).repeat(4, -1)
                contours_img = Image.fromarray(contours, mode="RGBA")
                print ('panoptic:', np.unique(pan_pred), pan_pred.shape, np.array(sem_img).shape, np.array(img).shape)

                out = Image.blend(img, sem_img, 0.5).convert(mode="RGBA")
                out = Image.alpha_composite(out, contours_img)
                cv2.imshow('pan_img', np.uint8(np.array(out.convert(mode="RGB"))))
                cv2.waitKey(0)

#                out.convert(mode="RGB").save(out_path)

            else:
                pan_uint = np.ones((pan_pred_.shape[0], pan_pred_.shape[1]), dtype=np.uint16)*255
                out_path = os.path.join(out_dir, outputFileName)
                #sem_ = sem_pred_.numpy()
                mask_ = np.logical_and(sem_>=17, sem_!=255)
                #print (mask_.shape)
                pan_uint[sem_<17] = sem_[sem_<17]*1000
                #print ('l', pan_uint.shape)
                pan_uint[mask_] = pan_pred_[mask_] + sem_[mask_] * 1000 + 1
                print (np.unique(pan_uint))
                pan_uint_ = np.zeros((pan_pred_.shape[0], pan_pred_.shape[1], 3), dtype=np.uint8)
                #print (np.unique(pan_uint))
                for k in np.unique(pan_uint):
                    if k == 255:
                       pan_uint_[pan_uint==k,2] = 255
                       continue

                    sem = map[k//1000]
                    inst = k%1000
#                    print (k, sem, inst)
                    pan_uint_[pan_uint==k,2] = sem
                    inst_g = inst // 256
                    inst_b = inst % 256
                    pan_uint_[pan_uint==k,1] = inst_g
                    pan_uint_[pan_uint==k,0] = inst_b

#                pred = pan_uint_
#                print(np.unique(pred[:,:,2]*1000 + (pred[:,:,1]*256) + (pred[:,:,0])))
                    #pan_uint_[pan_uint==k] = map[sem]*1000+inst
                #print (np.unique(pan_uint_))
                #for i, sem in enumerate(cat_pred.numpy()):
                #    if i == 0:
                #        continue
                #    pan_uint[pan_pred == i] = sem * 1000 + i
                cv2.imwrite(out_path, pan_uint_)

            prog_bar.update()

if __name__ == '__main__':
    main()
