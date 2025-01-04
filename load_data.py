# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
Author: Jingyu Liu
Date: 2024-01-04 15:52:12
LastEditTime: 2024-05-04 18:56:28
Description:
"""

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from math import ceil
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_util import get_video_text_align_train, get_video_text_align_val, get_cut_video

import pdb

class VDSFX_Dataset(Dataset):
    def __init__(self, sfx_path, sfx_text_feat_path, sfx_ast_feat_path, key_moment_path, video_name_txt,
                 video_tts_base, video_asr_base, frame_feat_base, tts_feat_base, asr_feat_base,
                 notext_np, query_num, nega_num, limit_frame_num, split):
        super(VDSFX_Dataset, self).__init__()

        # SFX_inform
        self.sfx_info = dict()
        sfx_id_list = [str(i) for i in list(pd.read_csv(sfx_path, encoding='utf-8')["id"])]
        sfx_tag_list = [str(i) for i in list(pd.read_csv(sfx_path, encoding='utf-8')["tag"])]
        sfx_tagid_list = [str(i) for i in list(pd.read_csv(sfx_path, encoding='utf-8')["tag_index"])]
        sfx_desc_list = [str(i) for i in list(pd.read_csv(sfx_path, encoding='utf-8')["description"])]
        for i in range(len(sfx_id_list)):
            sfx_id = sfx_id_list[i]
            sfx_tag = sfx_tag_list[i]
            sfx_tagid = int(sfx_tagid_list[i])
            sfx_desc = sfx_desc_list[i]
            sfx_ast_np = np.expand_dims(np.load(os.path.join(sfx_ast_feat_path, sfx_id + '.npy')), axis=0)
            sfx_text_np = np.expand_dims(np.load(os.path.join(sfx_text_feat_path, sfx_id + '.npy')), axis=0)
            self.sfx_info[sfx_id] = (sfx_desc, sfx_ast_np, sfx_text_np, sfx_tag, sfx_tagid)

        # moment
        moment_video = [str(i) for i in list(pd.read_csv(key_moment_path, encoding='utf-8')["name"])]
        moment_sfxid = [str(i) for i in list(pd.read_csv(key_moment_path, encoding='utf-8')["sfx_id"])]
        moment_frameid = [str(i) for i in list(pd.read_csv(key_moment_path, encoding='utf-8')["frameids"])]
        moment_info_list = []
        for i in range(len(moment_video)):
            moment_info_list.append((moment_video[i], moment_sfxid[i], moment_frameid[i]))

        # video
        video_list = []
        with open(video_name_txt, 'r', encoding="utf-8") as file:
            for line in file:
                if len(line.strip()) != 0:
                    video_list.append(line.strip())

        # frame
        self.frame_dict = dict()
        for name in video_list:
            frame_np_dir = os.path.join(frame_feat_base, name)
            frame_num = len(os.listdir(frame_np_dir))
            frame_feat_list = []
            for i in range(frame_num):
                feat_name = str(i).zfill(5) + '.npy'
                feat_np_path = os.path.join(frame_np_dir, feat_name)
                frame_feat_list.append(feat_np_path)
            self.frame_dict[name] = frame_feat_list

        # tts
        self.tts_dict = dict()
        for name in video_list:
            tts_info = os.path.join(video_tts_base, name + '.csv')
            if os.path.exists(tts_info):
                frame_num = len(self.frame_dict[name])
                tts_feat_list = []
                content_list = [str(i) for i in list(pd.read_csv(tts_info, encoding='utf-8')["content"])]
                start_list = [str(i) for i in list(pd.read_csv(tts_info, encoding='utf-8')["start"])]
                end_list = [str(i) for i in list(pd.read_csv(tts_info, encoding='utf-8')["end"])]
                feat_list = [str(i) for i in list(pd.read_csv(tts_info, encoding='utf-8')["feature"])]
                for i in range(len(content_list)):
                    if int(float(start_list[i])) >= frame_num:
                        continue
                    if int(float(end_list[i])) < 0:
                        continue
                    tts_content = content_list[i]
                    tts_start = max(int(float(start_list[i])), 0)
                    tts_end = min(int(float(end_list[i])), frame_num - 1)
                    tts_np_path = os.path.join(tts_feat_base, name, feat_list[i])
                    tts_feat_list.append((tts_content, tts_start, tts_end, tts_np_path))
                tts_feat_list = sorted(tts_feat_list, key=lambda list1: list1[1])
                if len(tts_feat_list) > 0:
                    self.tts_dict[name] = tts_feat_list

        # asr
        self.asr_dict = dict()
        for name in video_list:
            asr_info = os.path.join(video_asr_base, name + '.csv')
            if os.path.exists(asr_info):
                frame_num = len(self.frame_dict[name])
                asr_feat_list = []
                content_list = [str(i) for i in list(pd.read_csv(asr_info, encoding='utf-8')["content"])]
                start_list = [str(i) for i in list(pd.read_csv(asr_info, encoding='utf-8')["start"])]
                end_list = [str(i) for i in list(pd.read_csv(asr_info, encoding='utf-8')["end"])]
                feat_list = [str(i) for i in list(pd.read_csv(asr_info, encoding='utf-8')["feature"])]
                for i in range(len(content_list)):
                    if int(float(start_list[i])) >= frame_num:
                        continue
                    if int(float(end_list[i])) < 0:
                        continue
                    asr_content = content_list[i]
                    asr_start = max(int(float(start_list[i])), 0)
                    asr_end = min(int(float(end_list[i])), frame_num - 1)
                    asr_np_path = os.path.join(asr_feat_base, name, feat_list[i])
                    asr_feat_list.append((asr_content, asr_start, asr_end, asr_np_path))
                asr_feat_list = sorted(asr_feat_list, key=lambda list1: list1[1])
                if len(asr_feat_list) > 0:
                    self.asr_dict[name] = asr_feat_list

        # gather
        if split == 'train':
            self.video_final_dict = get_video_text_align_train(self.frame_dict, self.tts_dict, self.asr_dict,
                                                               moment_info_list)
        if split == 'val':
            self.video_final_dict = get_video_text_align_val(self.frame_dict, self.tts_dict, self.asr_dict,
                                                             moment_info_list, limit_frame_num, notext_np)

        self.split = split
        self.limit_frame_num = limit_frame_num
        self.notext_np = notext_np
        self.video_list = list(self.video_final_dict.keys())
        self.global_batch_size = 1
        self.sample_len = len(self.video_list)
        self.dataset_len = len(self.video_list)
        self.nega_num = nega_num
        self.query_num = query_num
    
    def __del__(self):
        pass

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.sample_len
        if self.split == 'train':
            video_info = get_cut_video(self.video_final_dict[self.video_list[sample_index]],
                                       self.limit_frame_num, self.notext_np)
        else:
            video_info = self.video_final_dict[self.video_list[sample_index]]

        # frame
        frame_feat_list = []
        for frame_feat_np_path in video_info[0]:
            frame_feat_list.append(torch.FloatTensor(np.expand_dims(np.load(frame_feat_np_path), axis=0)))
        frame_feat = torch.cat(frame_feat_list, 0)

        # tts
        tts_content_list = []
        tts_start_int_list = []
        tts_end_int_list = []
        tts_feat_list = []
        for tts_info in video_info[1]:
            tts_content_list.append(tts_info[0])
            tts_start_int_list.append(tts_info[1])
            tts_end_int_list.append(tts_info[2])
            tts_feat_list.append(torch.FloatTensor(np.expand_dims(np.load(tts_info[3]), axis=0)))
        tts_feat = torch.cat(tts_feat_list, 0)
        tts_list = [tts_feat, tts_start_int_list, tts_end_int_list]

        # asr
        asr_content_list = []
        asr_start_int_list = []
        asr_end_int_list = []
        asr_feat_list = []
        for asr_info in video_info[2]:
            asr_content_list.append(asr_info[0])
            asr_start_int_list.append(asr_info[1])
            asr_end_int_list.append(asr_info[2])
            asr_feat_list.append(torch.FloatTensor(np.expand_dims(np.load(asr_info[3]), axis=0)))
        asr_feat = torch.cat(asr_feat_list, 0)
        asr_list = [asr_feat, asr_start_int_list, asr_end_int_list]

        # moment
        sfx_id_list = []
        moment_frameid_list = []
        for sfx in video_info[3]:
            sfx_id = sfx[0]
            sfx_start = sfx[1]
            sfx_end = sfx[2]
            moment_frameid = list(range(sfx_start, sfx_end + 1))
            sfx_id_list.append(sfx_id)
            moment_frameid_list.append(moment_frameid)

        # background
        if len(sfx_id_list) == 0:
            back_num = self.query_num
        else:
            if len(sfx_id_list) < self.query_num:
                back_num = self.query_num - len(sfx_id_list)
            else:
                back_num = 0

        return frame_feat, tts_list, asr_list, sfx_id_list, moment_frameid_list, back_num


def custome_collate(batch):

    frame_batch_list = []
    tts_batch_list = []
    tts_sted_batch_list = []
    asr_batch_list = []
    asr_sted_batch_list = []

    frame_num_list = []
    frame_maxnum = -1
    tts_num_list = []
    tts_maxnum = -1
    asr_num_list = []
    asr_maxnum = -1

    sfx_id_batch_list = []
    moment_bbox_gt_list = []
    moment_frameid_batch_list = []
    moment_num_list = []
    back_num_list = []

    for frame_feat, tts_list, asr_list, sfx_id_list, moment_frameid_list, back_num in batch:
        # frame
        frame_batch_list.append(frame_feat)
        frame_num = frame_feat.shape[0]
        frame_num_list.append(frame_num)
        if frame_num > frame_maxnum:
            frame_maxnum = frame_num
        
        # tts
        tts_batch_list.append(tts_list[0])
        tts_num = tts_list[0].shape[0]
        tts_num_list.append(tts_num)
        if tts_num > tts_maxnum:
            tts_maxnum = tts_num
        tts_sted_per = []
        for i in range(tts_num):
            tts_sted_per.append((tts_list[1][i], tts_list[2][i]))
        tts_sted_batch_list.append(tts_sted_per)

        # asr
        asr_batch_list.append(asr_list[0])
        asr_num = asr_list[0].shape[0]
        asr_num_list.append(asr_num)
        if asr_num > asr_maxnum:
            asr_maxnum = asr_num
        asr_sted_per = []
        for i in range(asr_num):
            asr_sted_per.append((asr_list[1][i], asr_list[2][i]))
        asr_sted_batch_list.append(asr_sted_per)

        # moment
        for sfx_id, moment_frameid in zip(sfx_id_list, moment_frameid_list):
            sfx_start = max(moment_frameid[0] - 0.5, 0)
            sfx_end = min(moment_frameid[-1] + 0.5, frame_num)
            sfx_middle = (sfx_start + sfx_end) / (2 * frame_num)
            sfx_range = (sfx_end - sfx_start) / frame_num
            sfx_id_batch_list.append(sfx_id)
            moment_bbox_gt_list.append(torch.tensor([sfx_middle, sfx_range]).unsqueeze(0))
            moment_frameid_batch_list.append(moment_frameid)
        moment_num_list.append(len(sfx_id_list))
        back_num_list.append(back_num)
    
    # concat to batch ########################################################################################

    b = len(frame_batch_list)
    frame_dim = frame_batch_list[0].shape[-1]
    text_dim = tts_batch_list[0].shape[-1]
    # video feat
    frame_batch = torch.zeros(b, frame_maxnum, frame_dim)
    tts_batch = torch.zeros(b, tts_maxnum, text_dim)
    asr_batch = torch.zeros(b, asr_maxnum, text_dim)
    # position
    tts_start_pos = torch.zeros(b, tts_maxnum).int()
    tts_end_pos = torch.zeros(b, tts_maxnum).int()
    asr_start_pos = torch.zeros(b, asr_maxnum).int()
    asr_end_pos = torch.zeros(b, asr_maxnum).int()
    # mask
    frame_mask = -1000000 * torch.ones(b, frame_maxnum)
    tts_mask = -1000000 * torch.ones(b, tts_maxnum)
    asr_mask = -1000000 * torch.ones(b, asr_maxnum)

    for i in range(b):
        # video feat
        frame_batch[i, :frame_num_list[i]] = frame_batch_list[i]
        tts_batch[i, :tts_num_list[i]] = tts_batch_list[i]
        asr_batch[i, :asr_num_list[i]] = asr_batch_list[i]
        # position
        for j in range(tts_num_list[i]):
            tts_start_pos[i, j] = tts_sted_batch_list[i][j][0]
            tts_end_pos[i, j] = tts_sted_batch_list[i][j][1]
        for j in range(asr_num_list[i]):
            asr_start_pos[i, j] = asr_sted_batch_list[i][j][0]
            asr_end_pos[i, j] = asr_sted_batch_list[i][j][1]
        # mask
        frame_mask[i, :frame_num_list[i]] = torch.zeros(frame_num_list[i])
        tts_mask[i, :tts_num_list[i]] = torch.zeros(tts_num_list[i])
        asr_mask[i, :asr_num_list[i]] = torch.zeros(asr_num_list[i])

    moment_bbox_gt = torch.cat(moment_bbox_gt_list, 0)

    return frame_batch, tts_batch, asr_batch, \
           tts_start_pos, tts_end_pos, \
           asr_start_pos, asr_end_pos, \
           frame_mask, tts_mask, asr_mask, \
           frame_num_list, tts_num_list, asr_num_list, \
           sfx_id_batch_list, moment_bbox_gt, moment_frameid_batch_list, \
           moment_num_list, back_num_list

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: VDSFX_Dataset
    epoch_id: int


def get_traindata(args, dataset, is_train, epoch_id=0):
    batch_size = args.batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()

    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size

    num_samples = dataset.dataset_len

    sampler = DistributedSampler(dataset, rank=args.rank, num_replicas=args.world_size, shuffle=is_train, seed=args.seed)
    sampler.set_epoch(epoch_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=custome_collate,
        num_workers=args.num_workers if is_train else 2,
        sampler=sampler,
    )
    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_valdata(args, dataset, is_train):
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=custome_collate,
        num_workers=args.num_workers if is_train else 2,
        shuffle=False
    )

    return dataloader
