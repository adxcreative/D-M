# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
Author: Jingyu Liu
Date: 2024-01-04 15:52:12
LastEditTime: 2024-05-04 18:56:28
Description:
"""
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from util import get_sfx_nega, get_retrieval_rank

import pdb

class Statistic(object):
    def __init__(self):
        self.num = 0
        self.mean = 0
        self.cur = 0
    def add_one(self, val):
        self.mean = (self.mean*self.num+val) / (self.num+1)
        self.num += 1
        self.cur = val


def pretrain_epoch(model, data, epoch, optimizer, scheduler, args, global_trained_steps):
    model.train()
    dataloader, sampler = data.dataloader, data.sampler

    # loss
    criterion_sfx_match = nn.CrossEntropyLoss()

    statistic_tatal_loss = Statistic()
    statistic_msm_fore_loss = Statistic()
    statistic_msm_back_loss = Statistic()

    statistic_fore_top1_acc = Statistic()
    statistic_back_top1_acc = Statistic()

    # sfx
    sfx_info = data.dataset.sfx_info
    meta_sfx_id_list = []
    meta_sfx_ast_feat_list = []
    meta_sfx_text_feat_list = []
    meta_sfx_tagid_list = []
    for k, v in sfx_info.items():
        sfx_ast_feat = torch.tensor(v[1]).cuda(args.local_device_rank, non_blocking=True)
        sfx_text_feat = torch.tensor(v[2]).cuda(args.local_device_rank, non_blocking=True)
        sfx_tagid = torch.tensor([v[4]]).int().cuda(args.local_device_rank, non_blocking=True)
        meta_sfx_id_list.append(k)
        meta_sfx_ast_feat_list.append(sfx_ast_feat)
        meta_sfx_text_feat_list.append(sfx_text_feat)
        meta_sfx_tagid_list.append(sfx_tagid)
    meta_sfx_ast_feat = torch.cat(meta_sfx_ast_feat_list, dim=0)
    meta_sfx_text_feat = torch.cat(meta_sfx_text_feat_list, dim=0)
    meta_sfx_tagid = torch.cat(meta_sfx_tagid_list)

    if sampler is not None:
        sampler.set_epoch(epoch)
    num_steps_per_epoch = dataloader.num_batches
    data_iter = iter(dataloader)
    end = time.time()
    epoch_trained_steps = 0

    start_index = (global_trained_steps - num_steps_per_epoch * epoch) * args.accum_freq
    for i in range(start_index, dataloader.num_batches):
        batch = next(data_iter)
        i_accum = i
        step = num_steps_per_epoch * epoch + i_accum
        if step >= args.max_steps:
            return epoch_trained_steps
        scheduler(step)

        optimizer.zero_grad()

        frame_batch, tts_batch, asr_batch, \
        tts_start_pos, tts_end_pos, \
        asr_start_pos, asr_end_pos, \
        frame_mask, tts_mask, asr_mask, \
        frame_num_list, tts_num_list, asr_num_list, \
        sfx_id_batch_list, moment_bbox_gt, moment_frameid_batch_list, \
        moment_num_list, back_num_list = batch

        data_time = time.time() - end

        # cuda
        frame_batch = frame_batch.cuda(args.local_device_rank, non_blocking=True)
        tts_batch = tts_batch.cuda(args.local_device_rank, non_blocking=True)
        asr_batch = asr_batch.cuda(args.local_device_rank, non_blocking=True)
        tts_start_pos = tts_start_pos.cuda(args.local_device_rank, non_blocking=True)
        tts_end_pos = tts_end_pos.cuda(args.local_device_rank, non_blocking=True)
        asr_start_pos = asr_start_pos.cuda(args.local_device_rank, non_blocking=True)
        asr_end_pos = asr_end_pos.cuda(args.local_device_rank, non_blocking=True)
        frame_mask = frame_mask.cuda(args.local_device_rank, non_blocking=True)
        tts_mask = tts_mask.cuda(args.local_device_rank, non_blocking=True)
        asr_mask = asr_mask.cuda(args.local_device_rank, non_blocking=True)
        moment_bbox_gt = moment_bbox_gt.cuda(args.local_device_rank, non_blocking=True)

        # video feat
        video_feat, _ = model.module.encode_video(frame_batch, tts_batch, asr_batch,
                                               frame_mask, tts_mask, asr_mask,
                                               tts_start_pos, tts_end_pos,
                                               asr_start_pos, asr_end_pos)

        # moment-to-sfx matching
        moment_fore_feat_list = []
        moment_back_feat_list = []
        # get fore and back list of moment feature
        moment_num_count = 0
        for j, moment_num in enumerate(moment_num_list):
            frameid_for_fore = []
            if moment_num > 0:
                for k in range(moment_num_count, moment_num_count + moment_num):
                    frameid_th = torch.LongTensor(moment_frameid_batch_list[k]).cuda(args.local_device_rank, non_blocking=True)
                    moment_feat = torch.mean(video_feat[j][frameid_th], dim=0, keepdim=True)
                    moment_feat = moment_feat / moment_feat.norm(dim=-1, keepdim=True)
                    moment_fore_feat_list.append(moment_feat)
                    frameid_for_fore += moment_frameid_batch_list[k]
            frameid_for_fore = list(set(frameid_for_fore))
            frameid_for_back = []
            for k in range(frame_num_list[j]):
                if k not in frameid_for_fore:
                    frameid_for_back.append(k)
            if len(frameid_for_back) > 0:
                moment_feat = torch.mean(video_feat[j][torch.LongTensor(frameid_for_back).cuda(args.local_device_rank, non_blocking=True)], dim=0, keepdim=True)
                moment_feat = moment_feat / moment_feat.norm(dim=-1, keepdim=True)
                moment_back_feat_list.append(moment_feat)
            moment_num_count += moment_num

        # fore
        moment_batch_num = sum(moment_num_list)
        moment_fore_feat_batch = torch.cat(moment_fore_feat_list, dim=0)
        if moment_batch_num > 0:
            # nega sample
            sfx_choice_ids = []
            for sfx_id in sfx_id_batch_list:
                sfx_id_index = meta_sfx_id_list.index(sfx_id)
                sfx_nega_index = get_sfx_nega(sfx_id, meta_sfx_id_list, args.nega_num - 1)
                sfx_choice_ids.append([sfx_id_index] + sfx_nega_index)
            sfx_choice_ids_th = torch.LongTensor(sfx_choice_ids).cuda(args.local_device_rank, non_blocking=True)
            sfx_ast_feat_batch = meta_sfx_ast_feat[sfx_choice_ids_th]
            sfx_text_feat_batch = meta_sfx_text_feat[sfx_choice_ids_th]
            sfx_tagid_batch = meta_sfx_tagid[sfx_choice_ids_th]
            sfx_feat_batch_0 = model.module.encode_sfx(sfx_ast_feat_batch, sfx_text_feat_batch, sfx_tagid_batch)
            sfx_feat_batch = torch.cat((sfx_feat_batch_0, (model.module.sfx0 / model.module.sfx0.norm(dim=-1, keepdim=True))
                                        .unsqueeze(0).unsqueeze(0).repeat(moment_batch_num, 1, 1)), dim=1)
            # cosine-sim
            match_mat = model.module.logit_scale_sfx.exp() * torch.bmm(sfx_feat_batch,
                                                                       moment_fore_feat_batch.unsqueeze(2)).squeeze(2)
            match_gt = torch.zeros(match_mat.shape[0]).long().cuda(args.local_device_rank, non_blocking=True)
            fore_top1_acc = (match_mat.argmax(-1) == match_gt).sum() / match_mat.shape[0]
            fore_matching_loss = criterion_sfx_match(match_mat, match_gt)
        else:
            fore_top1_acc = torch.tensor(0).type_as(moment_fore_feat_batch).to(moment_fore_feat_batch.device)
            fore_matching_loss = torch.tensor(0).type_as(moment_fore_feat_batch).to(moment_fore_feat_batch.device)

        # back
        moment_back_feat_batch = torch.cat(moment_back_feat_list, dim=0)
        back_batch_num = moment_back_feat_batch.shape[0]
        if back_batch_num > 0:
            # nega sample
            sfx_choice_ids = []
            for _ in range(back_batch_num):
                sfx_nega_index = get_sfx_nega(None, meta_sfx_id_list, args.nega_num)
                sfx_choice_ids.append(sfx_nega_index)
            sfx_choice_ids_th = torch.LongTensor(sfx_choice_ids).cuda(args.local_device_rank, non_blocking=True)
            sfx_ast_feat_batch = meta_sfx_ast_feat[sfx_choice_ids_th]
            sfx_text_feat_batch = meta_sfx_text_feat[sfx_choice_ids_th]
            sfx_tagid_batch = meta_sfx_tagid[sfx_choice_ids_th]
            sfx_feat_batch_0 = model.module.encode_sfx(sfx_ast_feat_batch, sfx_text_feat_batch, sfx_tagid_batch)
            sfx_feat_batch = torch.cat(((model.module.sfx0 / model.module.sfx0.norm(dim=-1, keepdim=True))
                                        .unsqueeze(0).unsqueeze(0).repeat(back_batch_num, 1, 1),
                                        sfx_feat_batch_0), dim=1)
            # cosine-sim
            match_mat = model.module.logit_scale_sfx.exp() * torch.bmm(sfx_feat_batch,
                                                                       moment_back_feat_batch.unsqueeze(2)).squeeze(2)
            match_gt = torch.zeros(match_mat.shape[0]).long().cuda(args.local_device_rank, non_blocking=True)
            back_top1_acc = (match_mat.argmax(-1) == match_gt).sum() / match_mat.shape[0]
            back_matching_loss = criterion_sfx_match(match_mat, match_gt)
        else:
            back_top1_acc = torch.tensor(0).type_as(moment_back_feat_batch).to(moment_back_feat_batch.device)
            back_matching_loss = torch.tensor(0).type_as(moment_back_feat_batch).to(moment_back_feat_batch.device)

        # loss
        loss_all = args.pretrain_msm_fore * fore_matching_loss + \
                   args.pretrain_msm_back * back_matching_loss
        loss_all.backward()
        optimizer.step()

        model.module.logit_scale_sfx.data = torch.clamp(model.module.logit_scale_sfx.data, 0, 4.6052)
        batch_time = time.time() - end
        end = time.time()
        epoch_trained_steps += 1
        batchsize = video_feat.shape[0]
        if args.rank == 0:
            batch_size = batchsize * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            # loss
            statistic_tatal_loss.add_one(loss_all.item())
            statistic_msm_fore_loss.add_one(fore_matching_loss.item())
            statistic_msm_back_loss.add_one(back_matching_loss.item())

            # acc
            statistic_fore_top1_acc.add_one(fore_top1_acc.item())
            statistic_back_top1_acc.add_one(back_top1_acc.item())

            print(
                f"Global Steps: {step + 1}/{args.max_steps} | " + '\n' +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {statistic_tatal_loss.cur:.6f}/{statistic_tatal_loss.mean:.6f} | " +
                f"Loss fore: {statistic_msm_fore_loss.cur:.6f}/{statistic_msm_fore_loss.mean:.6f} | " +
                f"Loss back: {statistic_msm_back_loss.cur:.6f}/{statistic_msm_back_loss.mean:.6f} | " +
                (f"Acc fore_top1: {fore_top1_acc.item() * 100:.3f} | ") +
                (f"Acc back_top1: {back_top1_acc.item() * 100:.3f} | ") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_ae_scale: {model.module.logit_scale_sfx.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

    return epoch_trained_steps, (statistic_tatal_loss.mean, statistic_msm_fore_loss.mean, statistic_msm_back_loss.mean)


def pretrain_eval(model, data, args):
    model.eval()
    dataloader, sampler = data.dataloader, data.sampler

    with torch.no_grad():

        # sfx
        sfx_info = data.dataset.sfx_info
        meta_sfx_id_list = []
        meta_sfx_feat_list = []
        for k, v in sfx_info.items():
            sfx_ast_feat = torch.tensor(v[1]).cuda(args.local_device_rank, non_blocking=True)
            sfx_text_feat = torch.tensor(v[2]).cuda(args.local_device_rank, non_blocking=True)
            sfx_tagid = torch.tensor([v[4]]).int().cuda(args.local_device_rank, non_blocking=True)
            sfx_feat = model.module.encode_sfx(sfx_ast_feat.unsqueeze(1), sfx_text_feat.unsqueeze(1),
                                               sfx_tagid.unsqueeze(1)).squeeze(1)
            meta_sfx_id_list.append(k)
            meta_sfx_feat_list.append(sfx_feat.cpu().float())
        meta_sfx_id_list.append('sfx0')
        sfx0_feat = (model.module.sfx0 / model.module.sfx0.norm(dim=-1, keepdim=True)).unsqueeze(0).cpu().float()
        meta_sfx_feat_list.append(sfx0_feat)
        meta_sfx_feat = torch.cat(meta_sfx_feat_list, dim=0)

        # video
        moment_feat_list = []
        gt_sfxid_list = []

        data_iter = iter(dataloader)
        for i in range(dataloader.num_batches):
            batch = next(data_iter)

            frame_batch, tts_batch, asr_batch, \
            tts_start_pos, tts_end_pos, \
            asr_start_pos, asr_end_pos, \
            frame_mask, tts_mask, asr_mask, \
            frame_num_list, tts_num_list, asr_num_list, \
            sfx_id_batch_list, moment_bbox_gt, moment_frameid_batch_list, \
            moment_num_list, back_num_list = batch

            # cuda
            frame_batch = frame_batch.cuda(args.local_device_rank, non_blocking=True)
            tts_batch = tts_batch.cuda(args.local_device_rank, non_blocking=True)
            asr_batch = asr_batch.cuda(args.local_device_rank, non_blocking=True)
            tts_start_pos = tts_start_pos.cuda(args.local_device_rank, non_blocking=True)
            tts_end_pos = tts_end_pos.cuda(args.local_device_rank, non_blocking=True)
            asr_start_pos = asr_start_pos.cuda(args.local_device_rank, non_blocking=True)
            asr_end_pos = asr_end_pos.cuda(args.local_device_rank, non_blocking=True)
            frame_mask = frame_mask.cuda(args.local_device_rank, non_blocking=True)
            tts_mask = tts_mask.cuda(args.local_device_rank, non_blocking=True)
            asr_mask = asr_mask.cuda(args.local_device_rank, non_blocking=True)
            moment_bbox_gt = moment_bbox_gt.cuda(args.local_device_rank, non_blocking=True)

            # video feat
            video_feat, _ = model.module.encode_video(frame_batch, tts_batch, asr_batch,
                                                   frame_mask, tts_mask, asr_mask,
                                                   tts_start_pos, tts_end_pos,
                                                   asr_start_pos, asr_end_pos)

            # moment feature
            moment_num_count = 0
            for j, moment_num in enumerate(moment_num_list):
                if moment_num > 0:
                    for k in range(moment_num_count, moment_num_count + moment_num):
                        frameid_th = torch.LongTensor(moment_frameid_batch_list[k]).cuda(args.local_device_rank,
                                                                                         non_blocking=True)
                        moment_feat = torch.mean(video_feat[j][frameid_th], dim=0, keepdim=True)
                        moment_feat = moment_feat / moment_feat.norm(dim=-1, keepdim=True)
                        moment_feat_list.append(moment_feat.cpu().float())
                        gt_sfxid_list.append(sfx_id_batch_list[k])
                moment_num_count += moment_num

    # rank1
    moment_feat_all = torch.cat(moment_feat_list, dim=0)

    gt_sfx_index_list = []
    for sfxid in gt_sfxid_list:
        sfx_index = meta_sfx_id_list.index(sfxid)
        gt_sfx_index_list.append(sfx_index)
    match_mat = torch.mm(moment_feat_all, meta_sfx_feat.t())
    rank1 = get_retrieval_rank(match_mat, gt_sfx_index_list, 1)
    moment_num_all = match_mat.shape[0]
    print(rank1 / moment_num_all)

    rank1_th = torch.tensor([rank1]).cuda(args.local_device_rank, non_blocking=True)
    moment_num_all_th = torch.tensor([moment_num_all]).cuda(args.local_device_rank, non_blocking=True)
    dist.all_reduce(rank1_th, op=dist.ReduceOp.SUM)
    dist.all_reduce(moment_num_all_th, op=dist.ReduceOp.SUM)
    mean_rank1 = rank1_th.item() / moment_num_all_th.item()

    if args.rank == 0:
        print("Mean SFX Rank 1: %.3f" % mean_rank1)

    return mean_rank1

