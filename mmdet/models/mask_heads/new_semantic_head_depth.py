import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init
from math import ceil
import cv2
import numpy as np

from mmdet.core import auto_fp16, force_fp32
from mmdet.ops import ConvModule, DepthwiseSeparableConvModule, build_upsample_layer
from ..registry import HEADS

class Gate(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.r = 8
        #self.conv_kernel_size = 3
        #self.convs = nn.ModuleList()
        #dilations = [(1,3), (1,1), (2,7), (6,5), (2,1)]
        self.gap = nn.AdaptiveAvgPool2d(1)

        #for i in range(5):
        #    padding = dilations[i]
        #    self.convs.append(
        #        DepthwiseSeparableConvModule(
        #            self.in_channels,
        #            self.out_channels,
        #            self.conv_kernel_size,
        #            padding=padding,
        #            norm_cfg=norm_cfg,
        #            act_cfg=act_cfg,
        #            dilation=dilations[i]))
        #    self.in_channels = out_channels
        self.convr = ConvModule(
                    self.in_channels,
                    self.in_channels//self.r,
                    1,
                    padding=0)
        self.conve = ConvModule(
                    self.in_channels//self.r,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    act_cfg=None)

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.gap(x)
        x = self.convr(x)
        x = self.conve(x)
        x = self.act(x)
        return x


class MC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        x = F.interpolate(x, size=(x.shape[-2]*2, x.shape[-1]*2), 
                          mode='bilinear', align_corners=False)

        return x


class LSFE(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        for i in range(2):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                DepthwiseSeparableConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = self.convs(x)
        return x


class DPC(torch.nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, act_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.conv_out_channels = out_channels
        self.conv_kernel_size = 3
        self.convs = nn.ModuleList()
        dilations = [(1,6), (1,1), (6,21), (18,15), (6,3)]

        for i in range(5):
            padding = dilations[i]
            self.convs.append(
                DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dilation=dilations[i]))
            self.in_channels = self.conv_out_channels   
        self.conv = ConvModule(
                    self.conv_out_channels*5,
                    self.conv_out_channels,
                    1,
                    padding=0,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)

    def forward(self, x):
        x = self.convs[0](x)
        x1 = self.convs[1](x)
        x2 = self.convs[2](x)
        x3 = self.convs[3](x)
        x4 = self.convs[4](x3)
        x = torch.cat([
            x,
            x1,
            x2,
            x3,
            x4 
            ], dim=1)

        x = self.conv(x) 
        return x



@HEADS.register_module
class NewSemanticHeadDepth(nn.Module):
    def __init__(self,
                 in_channels=1088,
                 conv_out_channels=128,
                 num_classes=183,
                 ignore_label=255,
                 loss_weight=1.0,
                 loss_weight_depth=0.1, 
                 depth_channels=[32, 64],
                 max_depth = 88,  
                 ohem = 0.25,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 aux_loss=False,
                 add_extra_proposal=False,
                 instance_feat=False,
                 reverse_classes=False):

        super(NewSemanticHeadDepth, self).__init__()
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.loss_weight_depth = loss_weight_depth
        self.depth_channels = depth_channels
        self.max_depth = max_depth
        self.ohem = ohem
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.fp16_enabled = False
        self.aux_loss = aux_loss
        self.add_extra_proposal = add_extra_proposal
        self.instance_feat = instance_feat
        self.mse = nn.MSELoss()
        if self.ohem is not None:
            assert (self.ohem >= 0 and self.ohem < 1)

        #self.lateral_convs_ss = nn.ModuleList()
        self.lateral_convs_ls = nn.ModuleList()
        self.aligning_convs = nn.ModuleList()
        self.ss_idx = [3,2]
        self.ls_idx = [1,0] 
        self.dpc = DPC(
                    self.in_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        if self.instance_feat:
            self.up1 =  LSFE(
                    self.conv_out_channels+32+8,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
#            self.gate = Gate(8, 8, norm_cfg=self.norm_cfg,
#                    act_cfg=self.act_cfg)
        else:
            self.up1 =  LSFE(
                    self.conv_out_channels+32,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.up2 =  LSFE(
                    self.conv_out_channels+16,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.l1 = ConvModule(
                256,
                32,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

        self.l2 = ConvModule(
                256,
                16,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

#        for i in range(2):
#            self.lateral_convs_ls.append(

#        for i in range(2):
#            self.aligning_convs.append(
#                  MC(
#                    self.conv_out_channels,
#                    self.conv_out_channels,
#                    norm_cfg=self.norm_cfg,
#                    act_cfg=self.act_cfg))

        self.sem_conv =  LSFE(
                    self.conv_out_channels,
                    self.conv_out_channels,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.depth_conv1 =  LSFE(
                    self.conv_out_channels,
                    self.depth_channels[0],
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.depth_conv2 =  LSFE(
                    self.depth_channels[0],
                    self.depth_channels[1],
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

        self.conv_sem_logits = nn.Conv2d(conv_out_channels, self.num_classes, 1)
        self.conv_depth_logits = nn.Conv2d(self.depth_channels[1], 1, 1)
        if self.aux_loss:
            self.aux_conv_logits1 = nn.Conv2d(conv_out_channels, self.num_classes, 1)
            self.aux_conv_logits2 = nn.Conv2d(conv_out_channels, self.num_classes, 1)

        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')

    def init_weights(self):
        kaiming_init(self.conv_depth_logits)
        kaiming_init(self.conv_sem_logits)

    def forward(self, feats, side_feats, final_feat=None):
        #feats = list(feats)
        ref_size = tuple(side_feats[1].shape[-2:])
       
        x_ = self.dpc(feats)
        x = F.interpolate(x_, size=ref_size, 
                      mode='bilinear', align_corners=False)
        if self.aux_loss:
            a1 = self.aux_conv_logits1(x)
            a1 = F.interpolate(
                      a1, size=(ref_size[0]*8, ref_size[1]*8), 
                      mode='bilinear', align_corners=False)

        s1 = self.l1(side_feats[1])
        if final_feat is not None:
#            print ('shape:', final_feat.shape, s1.shape, x.shape)
#            final_feat = self.gate(final_feat)*final_feat
            x = torch.cat([x,s1, final_feat], dim=1) 
        else:
            x = torch.cat([x,s1], dim=1)
        x = self.up1(x)
        
        ref_size = tuple(side_feats[0].shape[-2:])

        x = F.interpolate(x, size=ref_size, 
                      mode='bilinear', align_corners=False)
        if self.aux_loss:
            a2 = self.aux_conv_logits2(x)
            a2 = F.interpolate(
                a2, size=(ref_size[0]*4, ref_size[1]*4), 
                mode='bilinear', align_corners=False)
        

        s1 = self.l2(side_feats[0])
        x = torch.cat([x,s1], dim=1)
        x_main = self.up2(x)
        # depth part
        xd = self.depth_conv1(x_main)
        xd = F.interpolate(xd, size=(ref_size[0]*2, ref_size[1]*2),
                      mode='bilinear', align_corners=False)
        xd = self.depth_conv2(xd) 
        xd = self.conv_depth_logits(xd)
        xd = F.interpolate(xd, size=(ref_size[0]*4, ref_size[1]*4), 
                      mode='bilinear', align_corners=False)

        # sem part
        x = self.sem_conv(x_main)
        x = self.conv_sem_logits(x)
        x = F.interpolate(
                      x, size=(ref_size[0]*4, ref_size[1]*4), 
                      mode='bilinear', align_corners=False)
        if self.aux_loss:
            return [x, a1, a2]
        if self.add_extra_proposal:
            return x, x_
        return x, xd

    def depth_loss(self, mask_pred, labels):
        loss = dict()
        mask_pred = self.max_depth * mask_pred.sigmoid()
        mask_pred = mask_pred.squeeze(1)
        mask_flag = labels != 0 # 0 is for ignore, make it more general later
#        print ('t', torch.sum(mask_flag[0, ...]), np.uint16(labels.cpu().numpy()).dtype)
#        cv2.imshow('img', np.uint16(labels[0,...].cpu().numpy()))
#        cv2.waitKey(0)
        labels = labels.float()/256.
        n = torch.sum(mask_flag)
        mse = self.mse(torch.log(labels[mask_flag]), torch.log(mask_pred[mask_flag])) 
        rel_err_a = ((torch.log(labels[mask_flag]) - torch.log(mask_pred[mask_flag])).sum() ** 2) / (n ** 2)
        rel_err_b = torch.sqrt((((labels[mask_flag] - mask_pred[mask_flag])/ labels[mask_flag]) ** 2).mean())
        depth_loss = mse + rel_err_b - rel_err_a
        loss['loss_depth'] = depth_loss
        return loss

    def depth_attention_loss(self, mask_pred, labels):
        loss = dict()
        mask_pred = self.max_depth * mask_pred.sigmoid()
        mask_pred = mask_pred.squeeze(1)
        mask_flag = labels != 0 # 0 is for ignore, make it more general later
        labels = labels.float() / 256.
        n = torch.sum(mask_flag)

        log_pred = torch.log(mask_pred[mask_flag])
        log_gt = torch.log(labels[mask_flag])

        a = labels[mask_flag] / self.max_depth # depth-aware attention
        reg = 1 - torch.minimum(log_pred, log_gt) / torch.maximum(log_pred, log_gt) # regularizer
        a_reg = a + reg

        l1 = ((a_reg * (log_gt - log_pred)) ** 2).mean()
        l2 = -((a_reg * (log_gt - log_pred)).mean() ** 2)
        l3 = (a_reg * (((labels[mask_flag] - mask_pred[mask_flag]) / labels[mask_flag]) ** 2)).mean().sqrt()
        depth_loss = l1 + l2 + l3
        loss['loss_depth'] = depth_loss
        return loss

    def loss(self, mask_pred, labels):
        loss = dict()
        labels = labels.squeeze(1).long()
        if self.aux_loss:
            aux_loss1 = self.criterion(mask_pred[1], labels)
            aux_loss1 = self.ohem_loss(aux_loss1, 0.5)
            aux_loss2 = self.criterion(mask_pred[2], labels)
            aux_loss2 = self.ohem_loss(aux_loss2, 0.6)
            loss_semantic_seg = self.criterion(mask_pred[0], labels)
            loss_semantic_seg = self.ohem_loss(loss_semantic_seg, self.loss_weight)
            loss_semantic_seg = loss_semantic_seg + aux_loss1 + aux_loss2                        
        else:
            loss_semantic_seg = self.criterion(mask_pred, labels)
            loss_semantic_seg = self.ohem_loss(loss_semantic_seg, self.loss_weight)

        loss['loss_semantic_seg'] = loss_semantic_seg
        return loss

    def ohem_loss(self, loss_semantic_seg, loss_weight):
        loss_semantic_seg = loss_semantic_seg.view(-1)
        if self.ohem is not None:
            top_k = int(ceil(loss_semantic_seg.numel() * self.ohem))
            if top_k != loss_semantic_seg.numel():
                    loss_semantic_seg, _ = loss_semantic_seg.topk(top_k)
 
        loss_semantic_seg = loss_semantic_seg.mean()
        loss_semantic_seg *= loss_weight
        return loss_semantic_seg

