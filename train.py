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
from train_util import train_epoch, evaluate_vdsfx, evaluate_kmd
from load_data import get_traindata, get_valdata, VDSFX_Dataset
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
    model.load_state_dict(torch.load(args.pretrain_model_path))
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
    val_data = get_valdata(args, val_dataset, False)

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

    train_loss = np.zeros((args.max_epochs, 5))
    val_vdsfx = np.zeros((args.max_epochs, 4))

    # pretrain
    for epoch in range(args.max_epochs):
        num_steps_this_epoch, train_epoch_loss = train_epoch(model, train_data, epoch, optimizer, scheduler, args, steps)
        if args.rank == 0:
            train_loss[epoch] = train_epoch_loss
            np.save(args.train_loss_save_path, train_loss)

        if epoch % args.save_period == 0:
            valvdsfx = evaluate_vdsfx(model, val_data, val_dataset, args)
            val_vdsfx[epoch] = valvdsfx
            np.save(args.train_valvdsfx_save_path, val_vdsfx)
            if args.rank == 0:
                torch.save(model.module.state_dict(), os.path.join(args.train_model_base, f"model_{epoch}.pth"))

        steps += num_steps_this_epoch

        if epoch + 1 < args.max_epochs:
            train_data = get_traindata(args, train_dataset, True, epoch + 1)

    # final
    val_vdsfx_mean = np.mean(val_vdsfx, axis=1, keepdims=False)
    best_id = np.argmax(val_vdsfx_mean)
    shutil.copyfile(os.path.join(args.train_model_base, f"model_{best_id}.pth"), args.train_model_path)
    print("Train Done!")

    # test
    test_model = DM(args.frame_dim, args.text_dim, args.ast_dim, args.hidden_dim,
                    args.encoder_layer_num, args.decoder_layer_num, args.head_num, args.att_dim,
                    args.att_dropout, args.ffn_dropout, args.query_num)
    test_model.load_state_dict(torch.load(args.train_model_path))
    test_model.cuda(args.local_device_rank)
    test_model = torch.nn.parallel.DistributedDataParallel(test_model, device_ids=[args.local_device_rank])
    test_dataset = VDSFX_Dataset(args.sfx_path, args.sfx_text_feat_path, args.sfx_ast_feat_path,
                                 args.test_key_moment_path, args.test_video_name_txt,
                                 args.test_tts_base, args.test_asr_base,
                                 args.test_frame_feat_base, args.test_tts_feat_base, args.test_asr_feat_base,
                                 args.notext_np, args.query_num, args.nega_num, args.limit_frame_num,
                                 'val')
    test_data = get_valdata(args, test_dataset, False)
    _ = evaluate_vdsfx(test_model, test_data, test_dataset, args)
    _ = evaluate_kmd(test_model, test_data, test_dataset, args)
    print("Test Done!")

if __name__ == "__main__":
    main()
