# !/usr/bin/env python
# -*-coding:utf-8 -*-
# Last Change:  2023-01-16 16:28:26

"""
# Author     ：Jingyu Liu
# File       : data_vip_caption.py
# Time       ：2024/01/03 20:49
"""
import random
from math import ceil
import numpy as np
import torch

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def get_retrieval_rank(match_mat, gt_index_list, rank_k):
    top_k_indices = torch.topk(match_mat, k=rank_k, dim=1)[1]
    ranks = []
    for i in range(match_mat.shape[0]):
        target_indices = top_k_indices[i]
        true_index = gt_index_list[i]
        rank = np.where(target_indices.cpu().numpy() == true_index)[0]
        if len(rank) > 0:
            ranks.append(rank[0] + 1)

    return len(ranks)


def nms(sfx_list, threshold):
    sfx_list = sorted(sfx_list, key=lambda list1: -list1[2][0])
    new_sfx_list = []

    while len(sfx_list) > 0:
        if len(sfx_list) == 1:
            new_sfx_list.append(sfx_list[0])
            break
        else:
            new_sfx_list.append(sfx_list[0])

        next_list = []
        for sfx in sfx_list[1:]:
            if sfx[0][0] >= sfx_list[0][0][1] or sfx[0][1] <= sfx_list[0][0][0]:
                iou = 0
            else:
                a_and_b = min(sfx[0][1], sfx_list[0][0][1]) - max(sfx[0][0], sfx_list[0][0][0])
                a_or_b = sfx[0][1] - sfx[0][0] + sfx_list[0][0][1] - sfx_list[0][0][0] - a_and_b
                iou = a_and_b / a_or_b
            if iou < threshold:
                next_list.append(sfx)
        sfx_list = next_list

    return new_sfx_list


def get_sfx_nega(sfx_id, meta_sfx_id_list, nega_num):
    if sfx_id != None:
        nega_cand_index_list = []
        for i, cand_gra_id in enumerate(meta_sfx_id_list):
            if cand_gra_id != sfx_id:
                nega_cand_index_list.append(i)
        nega_index_list = random.sample(nega_cand_index_list, nega_num)
    else:
        nega_cand_index_list = list(range(len(meta_sfx_id_list)))
        nega_index_list = random.sample(nega_cand_index_list, nega_num)

    return nega_index_list


def tans_sfx_nega(sfx_id, sfx_tag, momemt_sim, sfx_sim,
                  meta_sfx_id_list, meta_sfx_tag_list, sametag_num, nosame_num):
    sametag_index_list = []
    sametag_weight_list = []
    nosame_index_list = []
    nosame_weight_list = []

    nega_count_num = 0

    for i, cur_sfx_id in enumerate(meta_sfx_id_list):
        if cur_sfx_id != sfx_id and meta_sfx_tag_list[i] == sfx_tag:
            sametag_index_list.append(i)
            sametag_weight_list.append(sfx_sim[i])
            nega_count_num += 1
        if cur_sfx_id != sfx_id and meta_sfx_tag_list[i] != sfx_tag:
            nosame_index_list.append(i)
            nosame_weight_list.append(sfx_sim[i])

    if nega_count_num >= sametag_num:
        sametag_weight_np = np.array(sametag_weight_list) - momemt_sim
        sametag_weight_np = np.where(sametag_weight_np >= 0, -sametag_weight_np, sametag_weight_np)
        sametag_weight_np = sametag_weight_np - np.min(sametag_weight_np)
        sametag_weight = sametag_weight_np.tolist()
        sfx_sametag_nega_index = random.choices(sametag_index_list, weights=sametag_weight, k=sametag_num)
        nega_minus_num = 0
    else:
        sfx_sametag_nega_index = sametag_index_list
        nega_minus_num = sametag_num - nega_count_num

    nosame_num = nosame_num + nega_minus_num
    nosame_weight_np = np.array(nosame_weight_list)
    nosame_weight_np = np.where(nosame_weight_np > 0.05, nosame_weight_np, 0.05)
    nosame_weight = nosame_weight_np.tolist()
    sfx_nosame_nega_index = random.choices(nosame_index_list, weights=nosame_weight, k=nosame_num)

    return sfx_sametag_nega_index + sfx_nosame_nega_index
