# !/usr/bin/env python
# -*-coding:utf-8 -*-

import torch.nn.functional as F
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb


class HungarianMatcher(nn.Module):
    def __init__(self, HM_match = 1, HM_l1 = 1, HM_giou = 1):
        super().__init__()
        self.HM_match = HM_match
        self.HM_l1 = HM_l1
        self.HM_giou = HM_giou

    @torch.no_grad()
    def forward(self, moment_emb, moment_bbox, sfx_feat_gt, bbox_gt):
        cost_match = 1 - torch.mm(moment_emb, sfx_feat_gt.transpose(0, 1))
        cost_l1 = torch.cdist(moment_bbox, bbox_gt, p=1)
        giou = generalized_box_iou(moment_bbox, bbox_gt)
        cost_giou = - giou

        # cost matrix
        coss = self.HM_match * cost_match + self.HM_l1 * cost_l1 + self.HM_giou * cost_giou
        indices = linear_sum_assignment(coss.cpu())

        return torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)


def generalized_box_iou(boxes1, boxes2):
    start1 = boxes1[:, 0] - boxes1[:, 1] / 2
    start2 = boxes2[:, 0] - boxes2[:, 1] / 2
    end1 = boxes1[:, 0] + boxes1[:, 1] / 2
    end2 = boxes2[:, 0] + boxes2[:, 1] / 2
    lt = torch.max(start1[:, None], start2)
    rb = torch.min(end1[:, None], end2)
    inter = (rb - lt).clamp(min=0)
    union = boxes1[:, 1][:, None] + boxes2[:, 1] - inter
    iou = torch.where(union > 0,
                      inter / union,
                      torch.zeros_like(inter / union))

    area = (torch.max(end1[:, None], end2) - torch.min(start1[:, None], start2)).clamp(min=0)
    giou = iou - (area - union) / area

    return giou


def get_loss(matcher, moment_emb_list, moment_bbox_list,
             fore_sfx_feat_batch, back_sfx_feat_batch, moment_bbox_gt,
             moment_num_list, back_num_list, criterion_sfx_match, logit_scale_sfx, args):
    
    layer_loss_list = []
    fore_match_loss_list = []
    fore_l1_loss_list = []
    fore_giou_loss_list = []
    back_match_loss_list = []

    fore_top1_acc_list = []
    back_top1_acc_list = []

    layer_num = len(moment_emb_list)
    batch_size = moment_emb_list[0].shape[0]
    for i in range(layer_num):
        moment_num_count = 0
        fore_moment_emb_list = []
        fore_sfx_emb_list = []
        fore_moment_bbox_list = []
        fore_bbox_gt_list = []
        back_moment_emb_list = []
        for j in range(batch_size):
            moment_emb = moment_emb_list[i][j]
            moment_bbox = moment_bbox_list[i][j]
            moment_num = moment_num_list[j]
            # fore
            if moment_num > 0:
                sfx_feat_gt = fore_sfx_feat_batch[moment_num_count: moment_num_count + moment_num, 0]
                bbox_gt = moment_bbox_gt[moment_num_count: moment_num_count + moment_num]
                # Hungarian Matching
                indice_row, indice_col = matcher(moment_emb, moment_bbox, sfx_feat_gt, bbox_gt)
                # match
                fore_moment_emb = moment_emb[indice_row]
                fore_sfx_emb_gt = fore_sfx_feat_batch[moment_num_count: moment_num_count + moment_num][indice_col]
                fore_moment_emb_list.append(fore_moment_emb)
                fore_sfx_emb_list.append(fore_sfx_emb_gt)
                # bbox
                fore_moment_bbox = moment_bbox[indice_row]
                fore_bbox_gt = bbox_gt[indice_col]
                fore_moment_bbox_list.append(fore_moment_bbox)
                fore_bbox_gt_list.append(fore_bbox_gt)
                # back
                if back_num_list[j] > 0:
                    indice_row_list = indice_row.cpu().detach().tolist()
                    for k in range(args.query_num):
                        if k not in indice_row_list:
                            back_moment_emb_list.append(moment_emb[[k]])
            else:
                back_moment_emb_list.append(moment_emb)
            moment_num_count += moment_num
        
        # loss
        # fore
        if len(fore_moment_emb_list) > 0:
            # match
            fore_all_moment_emb = torch.cat(fore_moment_emb_list, dim=0)
            fore_all_sfx_emb = torch.cat(fore_sfx_emb_list, dim=0)
            match_mat = logit_scale_sfx.exp() * torch.bmm(fore_all_sfx_emb, fore_all_moment_emb.unsqueeze(2)).squeeze(2)
            match_gt = torch.zeros(match_mat.shape[0]).long().to(match_mat.device)
            fore_matching_loss = criterion_sfx_match(match_mat, match_gt)
            fore_top1_acc = (match_mat.argmax(-1) == match_gt).sum() / match_mat.shape[0]

            # bbox
            fore_all_moment_bbox = torch.cat(fore_moment_bbox_list, dim=0)
            fore_all_bbox_gt = torch.cat(fore_bbox_gt_list, dim=0)
            l1_loss = F.l1_loss(fore_all_moment_bbox, fore_all_bbox_gt)
            giou = generalized_box_iou(fore_all_moment_bbox, fore_all_bbox_gt)
            giou_loss = (1 - torch.diag(giou)).sum() / giou.shape[0]

        else:
            fore_matching_loss = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)
            fore_top1_acc = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)
            l1_loss = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)
            giou_loss = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)

        # back
        if len(back_moment_emb_list) > 0:
            # match
            back_all_moment_emb = torch.cat(back_moment_emb_list, dim=0)
            back_all_sfx_emb = back_sfx_feat_batch
            match_mat = logit_scale_sfx.exp() * torch.bmm(back_all_sfx_emb, back_all_moment_emb.unsqueeze(2)).squeeze(2)
            match_gt = torch.zeros(match_mat.shape[0]).long().to(match_mat.device)
            back_matching_loss = criterion_sfx_match(match_mat, match_gt)
            back_top1_acc = (match_mat.argmax(-1) == match_gt).sum() / match_mat.shape[0]
        else:
            back_matching_loss = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)
            back_top1_acc = torch.tensor(0).type_as(moment_bbox_gt).to(moment_bbox_gt.device)

        layer_loss = args.fore_match * fore_matching_loss + args.l1 * l1_loss + args.giou * giou_loss \
                     + args.back_match * back_matching_loss
        layer_loss_list.append(layer_loss)
        fore_match_loss_list.append(fore_matching_loss)
        fore_l1_loss_list.append(l1_loss)
        fore_giou_loss_list.append(giou_loss)
        back_match_loss_list.append(back_matching_loss)

        # acc
        fore_top1_acc_list.append(fore_top1_acc)
        back_top1_acc_list.append(back_top1_acc)

    return layer_loss_list, fore_match_loss_list, fore_l1_loss_list, fore_giou_loss_list, \
           back_match_loss_list, fore_top1_acc_list, back_top1_acc_list

    