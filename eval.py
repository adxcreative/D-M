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
import torch.distributed as dist

from model import DM
from train_util import evaluate_vdsfx, evaluate_kmd
from load_data import get_valdata, VDSFX_Dataset
from params import parse_args


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
    model.load_state_dict(torch.load(args.train_model_path))
    model.cuda(args.local_device_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_device_rank])

    # data
    test_dataset = VDSFX_Dataset(args.sfx_path, args.sfx_text_feat_path, args.sfx_ast_feat_path,
                                 args.test_key_moment_path, args.test_video_name_txt,
                                 args.test_tts_base, args.test_asr_base,
                                 args.test_frame_feat_base, args.test_tts_feat_base, args.test_asr_feat_base,
                                 args.notext_np, args.query_num, args.nega_num, args.limit_frame_num,
                                 'val')
    test_data = get_valdata(args, test_dataset, False)
    _ = evaluate_vdsfx(model, test_data, test_dataset, args)
    _ = evaluate_kmd(model, test_data, test_dataset, args)
    print("Evaluation Done!")

if __name__ == "__main__":
    main()
