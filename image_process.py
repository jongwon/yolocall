from train import main
import logging
from pathlib import Path
from threading import Thread
import numpy as np
import torch.distributed as dist
import torch, gc

import torch.utils.data
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils.torch_utils import ModelEMA, select_device

from utils.general import increment_path, check_file

from train_model import TrainingInfo
from utils.callbacks import Callbacks
import matplotlib as plt
from export import run
from utils.callbacks import Callbacks
import sys
import os

# plt.rc('axes', unicode_minus=False)

# logger = logging.getLogger(__name__)
# gc.collect()
# torch.cuda.empty_cache()

#// python train.py --batch 16 --img 640 --epochs 20 --data ./dataset/pistol/data.yaml --weights yolov5s.pt

class TrainArg():

    def __init__(self):
        self.batch_size = 8
        self.img_size = [640, 640]
        self.imgsz= 640
        self.epochs = 5
        self.data = './dataset/607e28967afab070717de46e/data.yaml'
        self.weights = 'yolov5s.pt'
        self.cfg = ''
        self.hyp = 'data/hyps/hyp.scratch.yaml'

        self.rect = True
        self.resume = False
        self.noval = False
        self.nosave = False
        self.notest = False
        self.noautoanchor = False
        self.evolve = False
        self.bucket = ''
        self.cache = 'ram'
        self.image_weights = False
        self.device = '0'
        self.multi_scale = False
        self.single_cls = False
        self.adam = False
        self.sync_bn = False
        self.local_rank = -1
        self.log_imgs = 16
        self.log_artifacts = False
        self.workers = 1
        self.project = 'runs/train'
        self.entity = None
        self.name=''
        self.exist_ok = True
        self.quad = False
        self.linear_lr = False
        self.cos_lr = False
        self.freeze = [0]
        self.optimizer = 'Adam'
        self.wandb= 'offline'
        self.upload_dataset= False
        self.bbox_interval= -1
        self.save_period= -1
        self.artifact_alias='latest'
        self.local_rank = 0
        self.patience = 100
        self.exist_ok=True
        self.label_smoothing=0.0
        # Set DDP variables
        self.world_size = 1
        self.global_rank = -1


def train_start(info: TrainingInfo):
    opt = TrainArg()
    opt.epochs=info.epochs
    opt.name=info.trainingId
    opt.batch_size=info.batchSize
    opt.img_size=[640, 640] #[info.imageSize, info.imageSize],
    opt.data=info.data
    opt.weights=info.weights
    opt.hyp=info.hyp
    print(vars(opt))
    train_models(opt, info.deviceForTorchscript)



def train_models(opt: TrainArg, device):
    # print(opt);

    # opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    # assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    #
    # opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    # opt.name = 'evolve' if opt.evolve else opt.name
    # opt.save_dir = str(Path(opt.project) / opt.name)  # increment run
    #
    # print(f'save dir {opt.save_dir}')
    #
    # # DDP mode
    # opt.total_batch_size = opt.batch_size
    # device = select_device(opt.device, batch_size=opt.batch_size)
    #
    # if opt.local_rank != -1:
    #     print('inside if')
    #     assert torch.cuda.device_count() > opt.local_rank
    #     torch.cuda.set_device(opt.local_rank)
    #     device = torch.device('cuda', opt.local_rank)
    #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
    #     opt.batch_size = opt.total_batch_size // opt.world_size
    # print('outside of if')
    #
    # # Hyperparameters
    # with open(opt.hyp) as f:
    #     print(f'load opt.hyp {opt.hyp}')
    #     hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # tb_writer = None  # init loggers
    # if opt.global_rank in [-1, 0]:
    #     logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
    #     tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
    # print(device, tb_writer, None)

    # dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
    # train(hyp, opt, device, Callbacks())
    main(opt)

    print('try opt.evolve if 지나감 1')
    run(include=('torchscript', 'onnx'), device=device,  weights=opt.save_dir+"/weights/best.pt")

    try:
        dist.destroy_process_group()
    except Exception as e:
        print('error(): '+ str(e))
        pass

    gc.collect()
    torch.cuda.empty_cache()

    print('try opt.evolve if 지나감 2')



    # print('try opt.evolve if 지나감 2 ')

if __name__ == '__main__':
    train_models()
