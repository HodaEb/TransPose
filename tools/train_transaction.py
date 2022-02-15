# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import numpy as np
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomVerticalFlip

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
import torch.nn as nn
from torch.nn import CrossEntropyLoss as cel
from core.function import train
from core.function import train_resnet
from core.function import train_transaction

from core.function import validate
from core.function import validate_transaction
from utils.utils import get_optimizer
from utils.utils import get_optimizer_resnet
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

# import dataset
# import models

import dataset as dataset
# import models as models
from models import transpose_h
from models import transpose_r
from models import transaction_h


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    seed = 22
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    #     cfg, is_train=True
    # )
    if cfg.MODEL.NAME == 'transpose_h':
        model = transpose_h.get_pose_net(cfg, is_train=True)
    elif cfg.MODEL.NAME == 'transaction_h':
        model = transaction_h.get_pose_net(cfg, is_train=True)
    else:
        model = transpose_r.get_pose_net(cfg, is_train=True)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    # writer_dict = {
    #     'writer': SummaryWriter(log_dir=tb_log_dir),
    #     'train_global_steps': 0,
    #     'valid_global_steps': 0,
    # }

    writer_dict = {
        'writer': SummaryWriter(log_dir='/content/drive/MyDrive/transaction_log/stanford/transaction_h'),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    trainable_parameters = []

    logger.info('doesnt need:')
    for name, param in model.named_parameters():
        # print(name)
        if 'action' in name:
            param.requires_grad = True
            trainable_parameters.append(param)
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, a=0, mode='fan_in', nonlinearity='relu')
        # elif 'apply_kp_htmp.weight' in name:
        #     param.requires_grad = True
        #     trainable_parameters.append(param)
        else:
            param.requires_grad = False
            logger.info(name)

    logger.info('needs:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
            logger.info(param)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = cel().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    average_acc = 0.0
    best_model = False
    last_epoch = -1

    # if 'resnet' in child:
    #     child.requires_grad = True
    # if not ('stage3' in child and 'fuselayers' in key):
    #     child.requires_grad = False

    # optimizer = get_optimizer(cfg, model)
    optimizer = get_optimizer_resnet(cfg, trainable_parameters)

    FINETUNE = cfg.MODEL.FINETUNE

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    if FINETUNE:
        initial_weights_file = cfg.TEST.MODEL_FILE
        initial_weights = torch.load(initial_weights_file)

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['accuracy_branch_1']
        # average_acc = checkpoint['accuracy_branch_1']
        last_epoch = checkpoint['epoch']

        writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        model.load_state_dict(checkpoint['state_dict'], strict=False)

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded last checkpoint '{}' (epoch {})".format(
            checkpoint_file, begin_epoch))
    else:
        logger.info("=> loading initial weights '{}'".format(initial_weights_file))

        model.module.load_state_dict(initial_weights, strict=False)  # strict=False FOR UNSeen Resolutions
        # logger.info("=> loaded checkpoint '{}' (epoch {})".format(
        #     checkpoint_file, checkpoint['epoch']))
        logger.info("=> loaded initial weights '{}' (epoch {})".format(
            initial_weights_file, begin_epoch))

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
    #     last_epoch=last_epoch
    # )

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch)

    model.cuda()

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        logger.info("=> current learning rate is {:.6f}".format(lr_scheduler.get_last_lr()[0]))
        # train for one epoch
        train_transaction(cfg, train_loader, model, criterion, optimizer, epoch,
                          final_output_dir, tb_log_dir, writer_dict)

        # average_acc_1, average_acc_2 = validate_transaction(cfg, valid_loader, valid_dataset, model, criterion,
        #                                                     final_output_dir, tb_log_dir)
        # if epoch % 4 == 0 :
        average_acc = validate_transaction(cfg, valid_loader, valid_dataset, model, criterion,
                                                            final_output_dir, tb_log_dir)

        lr_scheduler.step()
        # average_acc = (average_acc_1 + average_acc_2) / 2

        if average_acc >= best_perf:
            best_perf = average_acc
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'accuracy_branch_1': average_acc,
                # 'accuracy_branch_2': average_acc_2,
                'optimizer': optimizer.state_dict(),
                'train_global_steps': writer_dict['train_global_steps'],
                'valid_global_steps': writer_dict['valid_global_steps'],
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
