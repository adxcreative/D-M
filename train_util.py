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
import torch.nn.functional as F
import numpy as np
import math
from loss import HungarianMatcher, get_loss
from util import get_sfx_nega, tans_sfx_nega, nms
from eval_util import eval_vdsfx_sfx, eval_vdsfx_vid, eval_kmd_vid

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


def train_epoch(model, data, epoch, optimizer, scheduler, args, global_trained_steps):
    model.train()
    dataloader, sampler = data.dataloader, data.sampler

    # loss
    criterion_sfx_match = nn.CrossEntropyLoss()
    matcher = HungarianMatcher(args.HM_match, args.HM_l1, args.HM_giou)

    statistic_tatal_loss = Statistic()
    statistic_fore_match_loss = Statistic()
    statistic_fore_l1_loss = Statistic()
    statistic_fore_gious_loss = Statistic()
    statistic_back_match_loss = Statistic()

    statistic_fore_top1_acc = Statistic()
    statistic_back_top1_acc = Statistic()

    # sfx
    sfx_info = data.dataset.sfx_info
    meta_sfx_id_list = []
    meta_sfx_ast_feat_list = []
    meta_sfx_text_feat_list = []
    meta_sfx_tag_list = []
    meta_sfx_tagid_list = []
    for k, v in sfx_info.items():
        sfx_ast_feat = torch.tensor(v[1]).cuda(args.local_device_rank, non_blocking=True)
        sfx_text_feat = torch.tensor(v[2]).cuda(args.local_device_rank, non_blocking=True)
        sfx_tag = v[3]
        sfx_tagid = torch.tensor([v[4]]).int().cuda(args.local_device_rank, non_blocking=True)
        meta_sfx_id_list.append(k)
        meta_sfx_ast_feat_list.append(sfx_ast_feat)
        meta_sfx_text_feat_list.append(sfx_text_feat)
        meta_sfx_tag_list.append(sfx_tag)
        meta_sfx_tagid_list.append(sfx_tagid)
    meta_sfx_ast_feat = torch.cat(meta_sfx_ast_feat_list, dim=0)
    meta_sfx_text_feat = torch.cat(meta_sfx_text_feat_list, dim=0)
    meta_sfx_tagid = torch.cat(meta_sfx_tagid_list)

    # for TaNS
    sfx_tagdist_dict = dict()
    sfx_tagdist_dict['all'] = 0
    for sfx_tag in meta_sfx_tag_list:
        if sfx_tag not in sfx_tagdist_dict.keys():
            sfx_tagdist_dict[sfx_tag] = 0
        sfx_tagdist_dict[sfx_tag] += 1
        sfx_tagdist_dict['all'] += 1

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
        moment_emb_list, moment_bbox_list, video_feat = model.module.vdsfx(frame_batch, tts_batch, asr_batch,
                                                                           frame_mask, tts_mask, asr_mask,
                                                                           tts_start_pos, tts_end_pos,
                                                                           asr_start_pos, asr_end_pos)

        # moment-to-sfx matching
        moment_fore_feat_list = []
        moment_num_count = 0
        for j, moment_num in enumerate(moment_num_list):
            if moment_num > 0:
                for k in range(moment_num_count, moment_num_count + moment_num):
                    frameid_th = torch.LongTensor(moment_frameid_batch_list[k]).cuda(args.local_device_rank,
                                                                                     non_blocking=True)
                    moment_feat = torch.mean(video_feat[j][frameid_th], dim=0, keepdim=True)
                    moment_feat = moment_feat / moment_feat.norm(dim=-1, keepdim=True)
                    moment_fore_feat_list.append(moment_feat)
            moment_num_count += moment_num

        # fore
        moment_batch_num = sum(moment_num_list)
        moment_fore_feat_batch = torch.cat(moment_fore_feat_list, dim=0)
        if moment_batch_num > 0:
            # nega sample
            sfx_choice_ids = []
            if epoch < args.tans_start_epoch:
                # normal
                for sfx_id in sfx_id_batch_list:
                    sfx_id_index = meta_sfx_id_list.index(sfx_id)
                    sfx_nega_index = get_sfx_nega(sfx_id, meta_sfx_id_list, args.nega_num - 1)
                    sfx_choice_ids.append([sfx_id_index] + sfx_nega_index)
            else:
                # TaNS
                meta_sfx_feat = model.module.encode_sfx(meta_sfx_ast_feat, meta_sfx_text_feat, meta_sfx_tagid)
                moment2sfx_sim = np.array(torch.mm(moment_fore_feat_batch, meta_sfx_feat.t()).cpu().detach()).tolist()
                for j, sfx_id in enumerate(sfx_id_batch_list):
                    sfx_id_index = meta_sfx_id_list.index(sfx_id)
                    sfx_tag = meta_sfx_tag_list[sfx_id_index]
                    # moment_sim
                    momemt_sim = moment2sfx_sim[j][sfx_id_index]
                    sfx_sim = moment2sfx_sim[j]
                    # nega sample num from same tag and different tag
                    sametag_num = int((args.nega_num - 1) * sfx_tagdist_dict[sfx_tag] / sfx_tagdist_dict['all'])
                    sametag_num = min(args.nega_num - 1, sametag_num)
                    nosame_num = args.nega_num - 1 - sametag_num
                    # sample
                    sfx_nega_index = tans_sfx_nega(sfx_id, sfx_tag, momemt_sim, sfx_sim,
                                                   meta_sfx_id_list, meta_sfx_tag_list,
                                                   sametag_num, nosame_num)
                    sfx_choice_ids.append([sfx_id_index] + sfx_nega_index)
            sfx_choice_ids_th = torch.LongTensor(sfx_choice_ids).cuda(args.local_device_rank, non_blocking=True)
            sfx_ast_feat_batch = meta_sfx_ast_feat[sfx_choice_ids_th]
            sfx_text_feat_batch = meta_sfx_text_feat[sfx_choice_ids_th]
            sfx_tagid_batch = meta_sfx_tagid[sfx_choice_ids_th]
            sfx_feat_batch_0 = model.module.encode_sfx(sfx_ast_feat_batch, sfx_text_feat_batch, sfx_tagid_batch)
            fore_sfx_feat_batch = torch.cat((sfx_feat_batch_0, (model.module.sfx0 / model.module.sfx0.norm(dim=-1, keepdim=True))
                                        .unsqueeze(0).unsqueeze(0).repeat(moment_batch_num, 1, 1)), dim=1)
        else:
            fore_sfx_feat_batch = None

        # back
        sum_back_num = sum(back_num_list)
        if sum_back_num > 0:
            # nega sample
            sfx_choice_ids = []
            for _ in range(sum_back_num):
                sfx_nega_index = get_sfx_nega(None, meta_sfx_id_list, args.nega_num)
                sfx_choice_ids.append(sfx_nega_index)
            sfx_choice_ids_th = torch.LongTensor(sfx_choice_ids).cuda(args.local_device_rank, non_blocking=True)
            sfx_ast_feat_batch = meta_sfx_ast_feat[sfx_choice_ids_th]
            sfx_text_feat_batch = meta_sfx_text_feat[sfx_choice_ids_th]
            sfx_tagid_batch = meta_sfx_tagid[sfx_choice_ids_th]
            sfx_feat_batch_0 = model.module.encode_sfx(sfx_ast_feat_batch, sfx_text_feat_batch, sfx_tagid_batch)
            back_sfx_feat_batch = torch.cat(((model.module.sfx0 / model.module.sfx0.norm(dim=-1, keepdim=True))
                                        .unsqueeze(0).unsqueeze(0).repeat(sum_back_num, 1, 1),
                                        sfx_feat_batch_0), dim=1)
        else:
            back_sfx_feat_batch = None

        # loss
        layer_loss_list, fore_match_loss_list, fore_l1_loss_list, fore_giou_loss_list, \
        back_match_loss_list, fore_top1_acc_list, back_top1_acc_list = \
            get_loss(matcher, moment_emb_list, moment_bbox_list,
                     fore_sfx_feat_batch, back_sfx_feat_batch, moment_bbox_gt,
                     moment_num_list, back_num_list, criterion_sfx_match,
                     model.module.logit_scale_sfx, args)
        loss_all = 0
        layer_num = len(layer_loss_list)
        for j in range(layer_num):
            loss_all += layer_loss_list[j] / layer_num
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

            # show the performance of the last layer
            # loss
            statistic_tatal_loss.add_one(loss_all.item())
            statistic_fore_match_loss.add_one(fore_match_loss_list[-1].item())
            statistic_fore_l1_loss.add_one(fore_l1_loss_list[-1].item())
            statistic_fore_gious_loss.add_one(fore_giou_loss_list[-1].item())
            statistic_back_match_loss.add_one(back_match_loss_list[-1].item())
            # acc
            statistic_fore_top1_acc.add_one(fore_top1_acc_list[-1].item())
            statistic_back_top1_acc.add_one(back_top1_acc_list[-1].item())

            print(
                f"Global Steps: {step + 1}/{args.max_steps} | " + '\n' +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {statistic_tatal_loss.cur:.6f}/{statistic_tatal_loss.mean:.6f} | " +
                f"Loss fore_match: {statistic_fore_match_loss.cur:.6f}/{statistic_fore_match_loss.mean:.6f} | " +
                f"Loss fore_l1: {statistic_fore_l1_loss.cur:.6f}/{statistic_fore_l1_loss.mean:.6f} | " +
                f"Loss fore_giou: {statistic_fore_gious_loss.cur:.6f}/{statistic_fore_gious_loss.mean:.6f} | " +
                f"Loss back_match: {statistic_back_match_loss.cur:.6f}/{statistic_back_match_loss.mean:.6f} | " +
                (f"Acc fore_top1: {fore_top1_acc_list[-1].item() * 100:.3f} | ") +
                (f"Acc back_top1: {back_top1_acc_list[-1].item() * 100:.3f} | ") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_ae_scale: {model.module.logit_scale_sfx.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

    return epoch_trained_steps, (statistic_tatal_loss.mean, statistic_fore_match_loss.mean, statistic_fore_l1_loss.mean,
                                 statistic_fore_gious_loss.mean, statistic_back_match_loss.mean)


def evaluate_vdsfx(model, dataloader, dataset, args):
    model.eval()

    pred_video_all_list = []
    gt_video_all_list = []

    with torch.no_grad():

        # sfx
        sfx_info = dataset.sfx_info
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

        video_count = 0
        for i, batch in enumerate(dataloader):
            pred_video_list_batch = []
            gt_video_list_batch = []

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
            moment_emb_list, moment_bbox_list, _ = model.module.vdsfx(frame_batch, tts_batch, asr_batch,
                                                                               frame_mask, tts_mask, asr_mask,
                                                                               tts_start_pos, tts_end_pos,
                                                                               asr_start_pos, asr_end_pos)

            # inference
            batchsize = frame_batch.shape[0]
            for j in range(batchsize):
                pred_dict = {}
                pred_dict["qid"] = str(video_count + j)
                pred_dict["moment_pred_windows"] = []
                pred_dict["matched_sfx"] = []
                pred_dict["moment_score"] = []
                gt_dict = {}
                gt_dict["qid"] = str(video_count + j)
                gt_dict["moment_gt_windows"] = []
                gt_dict["gt_sfx"] = []
                pred_video_list_batch.append(pred_dict)
                gt_video_list_batch.append(gt_dict)

            # gt
            moment_num_count = 0
            for j, moment_num in enumerate(moment_num_list):
                if moment_num > 0:
                    for k in range(moment_num_count, moment_num_count + moment_num):
                        moment_gt_start = moment_frameid_batch_list[k][0]
                        moment_gt_end = moment_frameid_batch_list[k][-1] + 1
                        gt_sfx = sfx_id_batch_list[k]
                        gt_video_list_batch[j]["moment_gt_windows"].append([moment_gt_start, moment_gt_end])
                        gt_video_list_batch[j]["gt_sfx"].append(gt_sfx)
                moment_num_count += moment_num

            # pred
            for j in range(batchsize):
                pred_bbox = moment_bbox_list[-1][j].cpu()
                pred_bbox_sfx_score = []
                for k in range(args.query_num):
                    pred_bbox_start = math.ceil((pred_bbox[k][0] - pred_bbox[k][1] / 2) * frame_num_list[j])
                    pred_bbox_start = max(pred_bbox_start, 0)
                    pred_bbox_start = min(pred_bbox_start, frame_num_list[j] - 1)
                    pred_bbox_end = int((pred_bbox[k][0] + pred_bbox[k][1] / 2) * frame_num_list[j])
                    pred_bbox_end = max(pred_bbox_end, 0)
                    pred_bbox_end = min(pred_bbox_end, frame_num_list[j] - 1)
                    if pred_bbox_start > pred_bbox_end:
                        pred_bbox_start = pred_bbox_end
                    pred_bbox_end = pred_bbox_end + 1
                    # match
                    moment_emb = moment_emb_list[-1][j][k].cpu().float().unsqueeze(0)
                    match_mat = torch.mm(moment_emb, meta_sfx_feat.t())
                    match_mat = F.softmax(match_mat, dim=1)
                    # sfxid
                    top_indices = torch.topk(match_mat, k=10, dim=1)[1][0]
                    matched_sfxid_list = []
                    for kk in range(10):
                        matched_sfxid_list.append(meta_sfx_id_list[top_indices[kk]])
                    # moment_score
                    top_values = torch.topk(match_mat, k=10, dim=1)[0][0]
                    matched_scorelist = []
                    for kk in range(10):
                        matched_scorelist.append(top_values[kk].item())
                    if matched_sfxid_list[0] != 'sfx0':
                        pred_bbox_sfx_score.append([[pred_bbox_start, pred_bbox_end],
                                                    matched_sfxid_list, matched_scorelist])
                # nms
                pred_bbox_sfx_score = nms(pred_bbox_sfx_score, args.nms)
                for bbox, sfx, score in pred_bbox_sfx_score:
                    pred_video_list_batch[j]["moment_pred_windows"].append(bbox)
                    pred_video_list_batch[j]["matched_sfx"].append(sfx)
                    pred_video_list_batch[j]["moment_score"].append(score)

            video_count += batchsize
            pred_video_all_list += pred_video_list_batch
            gt_video_all_list += gt_video_list_batch

    top_list = [1]
    print('vdsfx_sfx_map')
    vdsfx_sfx_map = eval_vdsfx_sfx(pred_video_all_list, gt_video_all_list, top_list)
    for j, top_k in enumerate(top_list):
        print("map@0.5 top %d: %.3f" % (top_k, vdsfx_sfx_map[0, j]))
        print("map@0.75 top %d: %.3f" % (top_k, vdsfx_sfx_map[1, j]))

    print('vdsfx_vid_map')
    vdsfx_vid_map = eval_vdsfx_vid(pred_video_all_list, gt_video_all_list, top_list)
    for j, top_k in enumerate(top_list):
        print("map@0.5 top %d: %.3f" % (top_k, vdsfx_vid_map[0, j]))
        print("map@0.75 top %d: %.3f" % (top_k, vdsfx_vid_map[1, j]))

    return (vdsfx_sfx_map[0, 0], vdsfx_sfx_map[1, 0], vdsfx_vid_map[0, 0], vdsfx_vid_map[1, 0])


def evaluate_kmd(model, dataloader, dataset, args):
    model.eval()

    pred_video_all_list = []
    gt_video_all_list = []

    with torch.no_grad():

        # sfx
        sfx_info = dataset.sfx_info
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

        video_count = 0
        for i, batch in enumerate(dataloader):
            pred_video_list_batch = []
            gt_video_list_batch = []

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
            moment_emb_list, moment_bbox_list, _ = model.module.vdsfx(frame_batch, tts_batch, asr_batch,
                                                                               frame_mask, tts_mask, asr_mask,
                                                                               tts_start_pos, tts_end_pos,
                                                                               asr_start_pos, asr_end_pos)

            # inference
            batchsize = frame_batch.shape[0]
            for j in range(batchsize):
                pred_dict = {}
                pred_dict["qid"] = str(video_count + j)
                pred_dict["moment_pred_windows"] = []
                gt_dict = {}
                gt_dict["qid"] = str(video_count + j)
                gt_dict["moment_gt_windows"] = []
                pred_video_list_batch.append(pred_dict)
                gt_video_list_batch.append(gt_dict)

            # gt
            moment_num_count = 0
            for j, moment_num in enumerate(moment_num_list):
                if moment_num > 0:
                    for k in range(moment_num_count, moment_num_count + moment_num):
                        moment_gt_start = moment_frameid_batch_list[k][0]
                        moment_gt_end = moment_frameid_batch_list[k][-1] + 1
                        gt_video_list_batch[j]["moment_gt_windows"].append([moment_gt_start, moment_gt_end])
                moment_num_count += moment_num

            # pred
            for j in range(batchsize):
                pred_bbox = moment_bbox_list[-1][j].cpu()
                pred_bbox_sfx_score = []
                for k in range(args.query_num):
                    pred_bbox_start = math.ceil((pred_bbox[k][0] - pred_bbox[k][1] / 2) * frame_num_list[j])
                    pred_bbox_start = max(pred_bbox_start, 0)
                    pred_bbox_start = min(pred_bbox_start, frame_num_list[j] - 1)
                    pred_bbox_end = int((pred_bbox[k][0] + pred_bbox[k][1] / 2) * frame_num_list[j])
                    pred_bbox_end = max(pred_bbox_end, 0)
                    pred_bbox_end = min(pred_bbox_end, frame_num_list[j] - 1)
                    if pred_bbox_start > pred_bbox_end:
                        pred_bbox_start = pred_bbox_end
                    pred_bbox_end = pred_bbox_end + 1
                    # match
                    moment_emb = moment_emb_list[-1][j][k].cpu().float().unsqueeze(0)
                    match_mat = torch.mm(moment_emb, meta_sfx_feat.t())
                    match_mat = F.softmax(match_mat, dim=1)
                    # sfxid
                    top_indices = torch.topk(match_mat, k=10, dim=1)[1][0]
                    matched_sfxid_list = []
                    for kk in range(10):
                        matched_sfxid_list.append(meta_sfx_id_list[top_indices[kk]])
                    # moment_score
                    top_values = torch.topk(match_mat, k=10, dim=1)[0][0]
                    matched_scorelist = []
                    for kk in range(10):
                        matched_scorelist.append(top_values[kk].item())
                    pred_bbox_sfx_score.append([[pred_bbox_start, pred_bbox_end], matched_sfxid_list, matched_scorelist])
                # nms
                pred_bbox_sfx_score = nms(pred_bbox_sfx_score, args.nms)
                for bbox, sfx, score in pred_bbox_sfx_score:
                    pred_video_list_batch[j]["moment_pred_windows"].append([bbox[0], bbox[1], float(score[0])])

            video_count += batchsize
            pred_video_all_list += pred_video_list_batch
            gt_video_all_list += gt_video_list_batch

    print('kmd_vid_map')
    kmd_vid_map = eval_kmd_vid(pred_video_all_list, gt_video_all_list)
    print("map@0.5: %.3f" % kmd_vid_map[0])
    print("map@0.75: %.3f" % kmd_vid_map[1])

    return (kmd_vid_map[0], kmd_vid_map[1], kmd_vid_map[0], kmd_vid_map[1])

