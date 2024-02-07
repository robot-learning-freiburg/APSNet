import warnings

warnings.filterwarnings("ignore")
import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.core.utils import prepare_cityscapes_benchmarking


class MultipleKVAction(argparse.Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options should
    be passed as comma separated values, i.e KEY=V1,V2,V3
    """

    def _parse_int_float_bool(self, val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(",")]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("out", help="output root folder")
    parser.add_argument(
        "--task",
        type=str,
        help='task format, which depends on the dataset, e.g., "panoptic" for Cityscapes,'
        ' "amodal-panoptic", for amodal panoptic segmentation datasets',
    )
    parser.add_argument(
        "--split", action="store_true", default="val", help="show results"
    )
    parser.add_argument(
        "--gpu_collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu_collect is not specified",
    )
    parser.add_argument(
        "--options", nargs="+", action=MultipleKVAction, help="custom options"
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    assert args.task, "Please specify evaluation tasks, e.g., --task amodal-panoptic"

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    os.makedirs(args.out, exist_ok=True)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

        port = os.environ.get("MASTER_PORT", "default_port")
        if port is None:
            raise ValueError("MASTER_PORT is not set in the environment.")

    # build the dataloader
    dataset_type = cfg.dataset_type
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.val_imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get("fp16", None)

    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    if "CLASSES" in checkpoint["meta"]:
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(
            model,
            data_loader,
            args.out,
            eval=[
                args.task,
            ],
        )
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(
            model,
            data_loader,
            args.out,
            args.tmpdir,
            args.gpu_collect,
            [
                args.task,
            ],
        )

    rank, _ = get_dist_info()
    if rank == 0:
        if args.task == "panoptic" and dataset_type == "CityscapesDataset":
            prepare_cityscapes_benchmarking(args.out, args.split)


if __name__ == "__main__":
    main()
