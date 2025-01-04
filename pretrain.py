# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
Author: Jingyu Liu
Date: 2024-01-04 15:52:12
LastEditTime: 2024-05-04 18:56:28
Description:
"""
from math import ceil
import os
import numpy as np
import random
import torch
from torch import optim
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import shutil

from model import DM
from pretrain_util import pretrain_epoch, pretrain_eval
from load_data import get_traindata, VDSFX_Dataset
from params import parse_args
from util import cosine_lr


def main():
    args = parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args.local_device_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)
    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    # model
    model = DM(args.frame_dim, args.text_dim, args.ast_dim, args.hidden_dim,
               args.encoder_layer_num, args.decoder_layer_num, args.head_num, args.att_dim,
               args.att_dropout, args.ffn_dropout, args.query_num)
    model.cuda(args.local_device_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_device_rank])

    # data
    train_dataset = VDSFX_Dataset(args.sfx_path, args.sfx_text_feat_path, args.sfx_ast_feat_path,
                                  args.train_key_moment_path, args.train_video_name_txt,
                                  args.train_tts_base, args.train_asr_base,
                                  args.train_frame_feat_base, args.train_tts_feat_base, args.train_asr_feat_base,
                                  args.notext_np, args.query_num, args.nega_num, args.limit_frame_num,
                                  'train')

    val_dataset = VDSFX_Dataset(args.sfx_path, args.sfx_text_feat_path, args.sfx_ast_feat_path,
                                args.val_key_moment_path, args.val_video_name_txt,
                                args.val_tts_base, args.val_asr_base,
                                args.val_frame_feat_base, args.val_tts_feat_base, args.val_asr_feat_base,
                                args.notext_np, args.query_num, args.nega_num, args.limit_frame_num,
                                'val')

    train_data = get_traindata(args, train_dataset, True, epoch_id=0)
    val_data = get_traindata(args, val_dataset, False, epoch_id=0)

    # lr
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    num_batches = train_data.dataloader.num_batches
    if args.max_steps is not None:
        args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
    else:
        assert args.max_epochs is not None and args.max_epochs > 0
        args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
    total_steps = args.max_steps
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    steps = 0
    cudnn.benchmark = True
    cudnn.deterministic = False

    train_loss = np.zeros((args.max_epochs, 3))
    val_rank1 = np.zeros(args.max_epochs)

    # pretrain
    for epoch in range(args.max_epochs):
        num_steps_this_epoch, train_epoch_loss = pretrain_epoch(model, train_data, epoch, optimizer, scheduler, args, steps)
        if args.rank == 0:
            train_loss[epoch] = train_epoch_loss
            np.save(args.pretrain_loss_save_path, train_loss)

        if epoch % args.save_period == 0:
            valrank1 = pretrain_eval(model, val_data, args)
            val_rank1[epoch] = valrank1
            np.save(args.pretrain_valrank1_save_path, val_rank1)
            if args.rank == 0:
                torch.save(model.module.state_dict(), os.path.join(args.pretrain_model_base, f"model_{epoch}.pth"))

        steps += num_steps_this_epoch

        if epoch + 1 < args.max_epochs:
            train_data = get_traindata(args, train_dataset, True, epoch + 1)
            val_data = get_traindata(args, val_dataset, False, epoch + 1)

    # final
    best_id = np.argmax(val_rank1)
    shutil.copyfile(os.path.join(args.pretrain_model_base, f"model_{best_id}.pth"), args.pretrain_model_path)
    print("Pretrain Done!")

if __name__ == "__main__":
    main()
