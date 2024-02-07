import numpy as np
import pycls.models.blocks as bk
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from inplace_abn import ABN
from ..registry import BACKBONES
from mmdet.utils import get_root_logger

cfg = {'RegNetY-4.0GF': {'WA':31.41, 'WM':2.24, 'W0':96, 'DEPTH':22, 'SE':0, 'NUM_CLASSES':10, 'BLOCK_TYPE':'res_bottleneck_block',
                          'STEM_TYPE':'simple_stem_in', 'GROUP_W':64, 'BOT_MUL':1, 'STRIDE':2, 'STEM_W':32,},
       'RegNetY-32GF': {'WA':115.89, 'WM':2.53, 'W0':232, 'DEPTH':20, 'SE':0, 'NUM_CLASSES':10, 'BLOCK_TYPE':'res_bottleneck_block',
                          'STEM_TYPE':'simple_stem_in', 'GROUP_W':232, 'BOT_MUL':1, 'STRIDE':2, 'STEM_W':32,},
       'RegNetY-8.0GF': {'WA':76.82, 'WM':2.19, 'W0':192, 'DEPTH':17, 'SE':0, 'NUM_CLASSES':10, 'BLOCK_TYPE':'res_bottleneck_block',
                          'STEM_TYPE':'simple_stem_in', 'GROUP_W':56, 'BOT_MUL':1, 'STRIDE':2, 'STEM_W':32,},
       'RegNetY-16GF': {'WA':106.23, 'WM':2.48, 'W0':200, 'DEPTH':18, 'SE':0, 'NUM_CLASSES':10, 'BLOCK_TYPE':'res_bottleneck_block',
                          'STEM_TYPE':'simple_stem_in', 'GROUP_W':112, 'BOT_MUL':1, 'STRIDE':2, 'STEM_W':32,}}


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per stage widths and depths from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    # Generate continuous per-block ws
    ws_cont = np.arange(d) * w_a + w_0
    # Generate quantized per-block ws
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws_all = w_0 * np.power(w_m, ks)
    ws_all = np.round(np.divide(ws_all, q)).astype(int) * q
    # Generate per stage ws and ds (assumes ws_all are sorted)
    ws, ds = np.unique(ws_all, return_counts=True)
    # Compute number of actual stages and total possible stages
    num_stages, total_stages = len(ws), ks.max() + 1
    # Convert numpy arrays to lists and return
    ws, ds, ws_all, ws_cont = (x.tolist() for x in (ws, ds, ws_all, ws_cont))
    return ws, ds, num_stages, total_stages, ws_all, ws_cont

@BACKBONES.register_module
class RegNet(AnyNet):
    """RegNet model."""

    @staticmethod
    def get_params(cfg):
        """Convert RegNet to AnyNet parameter format."""
        # Generates per stage ws, ds, gs, bs, and ss from RegNet parameters
        w_a, w_0, w_m, d = cfg['WA'], cfg['W0'], cfg['WM'], cfg['DEPTH']
        ws, ds = generate_regnet(w_a, w_0, w_m, d)[0:2]
        ss = [cfg['STRIDE'] for _ in ws]
        bs = [cfg['BOT_MUL'] for _ in ws]
        gs = [cfg['GROUP_W'] for _ in ws]
        ws, bs, gs = bk.adjust_block_compatibility(ws, bs, gs)
        # Get AnyNet arguments defining the RegNet
        return {
            "stem_type": cfg['STEM_TYPE'],
            "stem_w": cfg['STEM_W'],
            "block_type": cfg['BLOCK_TYPE'],
            "depths": ds,
            "widths": ws,
            "strides": ss,
            "bot_muls": bs,
            "group_ws": gs,
            "se_r":  0,
            "num_classes": cfg['NUM_CLASSES'],
        }

    def __init__(self, name, with_extra=False):
        params = RegNet.get_params(cfg[name])
        super(RegNet, self).__init__(params, with_extra)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            model = self
            logger = get_root_logger()
            checkpoint = _load_checkpoint(pretrained, None)
            state_dict = checkpoint['model_state']
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in checkpoint['model_state'].items()}
            # load state_dict
            if hasattr(model, 'module'):
                load_state_dict(model.module, state_dict, strict, logger)
            else:
                load_state_dict(model, state_dict, False, logger)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, ABN)):
                    constant_init(m, 1)
                    
        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def complexity(cx, params=None):
        """Computes model complexity (if you alter the model, make sure to update)."""
        params = RegNet.get_params() if not params else params
        return AnyNet.complexity(cx, params)

