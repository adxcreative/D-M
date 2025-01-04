# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
Author: Jingyu Liu
Date: 2024-01-04 15:52:12
LastEditTime: 2024-05-04 18:56:28
Description:
"""

import random
import pdb


def cut_text_num(cut_video_text, notext_np, split):
    if split == 'train':
        if len(cut_video_text) > 40:
            text_choose_id = random.sample(list(range(len(cut_video_text))), 40)
            new_cut_video_text = []
            for hh in text_choose_id:
                new_cut_video_text.append(cut_video_text[hh])
            new_cut_video_text.append((' ', 0, 0, notext_np))
        elif len(cut_video_text) == 0:
            new_cut_video_text = [(' ', 0, 0, notext_np)]
        else:
            new_cut_video_text = cut_video_text
            new_cut_video_text.append((' ', 0, 0, notext_np))
    else:
        if len(cut_video_text) == 0:
            new_cut_video_text = [(' ', 0, 0, notext_np)]
        else:
            new_cut_video_text = cut_video_text
            new_cut_video_text.append((' ', 0, 0, notext_np))

    return new_cut_video_text


def cut_moment_num(cur_moment_list, split):
    if split == 'train':
        if len(cur_moment_list) > 10:
            sfx_choose_id = random.sample(list(range(len(cur_moment_list))), 10)
            new_cur_moment_list = []
            for sfx_id in sfx_choose_id:
                new_cur_moment_list.append(cur_moment_list[sfx_id])
        else:
            new_cur_moment_list = cur_moment_list
    else:
        new_cur_moment_list = cur_moment_list

    return new_cur_moment_list


def get_videocut_from_startend(video_info, cut_video_start, cut_video_end, notext_np, split):
    cut_video_frame = []
    cut_video_tts = []
    cut_video_asr = []
    cur_moment_list = []
    # frame
    for j, frame in enumerate(video_info.frame_list):
        if j >= cut_video_start and j < cut_video_end:
            cut_video_frame.append(frame)
    # tts
    for tts in video_info.tts_list:
        if tts[1] >= cut_video_end or tts[2] < cut_video_start:
            continue
        else:
            tts_content = tts[0]
            tts_start = max(tts[1], cut_video_start) - cut_video_start
            tts_end = min(tts[2], cut_video_end - 1) - cut_video_start
            tts_feat_np = tts[3]
            cut_video_tts.append((tts_content, tts_start, tts_end, tts_feat_np))
    # asr
    for asr in video_info.asr_list:
        if asr[1] >= cut_video_end or asr[2] < cut_video_start:
            continue
        else:
            asr_content = asr[0]
            asr_start = max(asr[1], cut_video_start) - cut_video_start
            asr_end = min(asr[2], cut_video_end - 1) - cut_video_start
            asr_feat_np = asr[3]
            cut_video_asr.append((asr_content, asr_start, asr_end, asr_feat_np))
    # moment
    for sfx in video_info.sfx_list:
        if sfx[1] >= cut_video_end or sfx[2] < cut_video_start:
            continue
        else:
            sfx_id = sfx[0]
            moment_start = max(sfx[1], cut_video_start) - cut_video_start
            moment_end = min(sfx[2], cut_video_end - 1) - cut_video_start
            cur_moment_list.append((sfx_id, moment_start, moment_end))

    if len(cur_moment_list) > 0:
        new_cut_video_tts = cut_text_num(cut_video_tts, notext_np, split)
        new_cut_video_asr = cut_text_num(cut_video_asr, notext_np, split)
        new_cur_moment_list = cut_moment_num(cur_moment_list, split)
        return (cut_video_frame, new_cut_video_tts, new_cut_video_asr, new_cur_moment_list)
    else:
        return None


def get_video_text_align_train(frame_dict, tts_dict, asr_dict, moment_info_list):
    class video:
        def __init__(self):
            self.frame_num = 0
            self.frame_list = []
            self.tts_list = []
            self.asr_list = []
            self.sfx_list = []
            self.moment_start = 1000
            self.moment_end = 0

    # video
    video_dict = dict()
    for k, v in frame_dict.items():
        if k not in video_dict.keys():
            video_dict[k] = video()
        video_dict[k].frame_num = len(v)
        video_dict[k].frame_list = v
    for k, v in tts_dict.items():
        video_dict[k].tts_list = v
    for k, v in asr_dict.items():
        video_dict[k].asr_list = v

    # moment
    for moment_info in moment_info_list:
        k = moment_info[0]
        if int(moment_info[2].split('*')[0]) >= video_dict[k].frame_num:
            continue
        if int(moment_info[2].split('*')[-1]) < 0:
            continue
        moment_start = max(int(moment_info[2].split('*')[0]), 0)
        moment_end = min(int(moment_info[2].split('*')[-1]), video_dict[k].frame_num - 1)
        video_dict[k].sfx_list.append((moment_info[1], moment_start, moment_end))
        video_dict[k].moment_start = min(moment_start, video_dict[k].moment_start)
        video_dict[k].moment_end = max(moment_end, video_dict[k].moment_end)

    # gather
    uncut_video_dict = dict()
    for k, v in video_dict.items():
        if v.frame_num == 0:
            continue
        if len(v.sfx_list) == 0:
            continue
        uncut_video_dict[k] = video()
        uncut_video_dict[k].frame_num = len(v.frame_list)
        uncut_video_dict[k].frame_list = v.frame_list
        uncut_video_dict[k].tts_list = v.tts_list
        uncut_video_dict[k].asr_list = v.asr_list
        uncut_video_dict[k].sfx_list = v.sfx_list
        uncut_video_dict[k].moment_start = v.moment_start
        uncut_video_dict[k].moment_end = v.moment_end

    return uncut_video_dict


def get_cut_video(uncut_video_info, limit_frame_num, notext_np):
    frame_num = uncut_video_info.frame_num
    if frame_num <= limit_frame_num:
        cut_video_start = 0
        cut_video_end = frame_num
        cutted_video = get_videocut_from_startend(uncut_video_info, cut_video_start, cut_video_end, notext_np, 'train')
    else:
        choice_pool = []
        for sfx in uncut_video_info.sfx_list:
            choice_pool += list(range(max(sfx[2] - limit_frame_num + 1, 0), sfx[1] + 1))
        cut_video_start = random.choice(list(set(choice_pool)))
        cut_video_end = cut_video_start + limit_frame_num
        cutted_video = get_videocut_from_startend(uncut_video_info, cut_video_start, cut_video_end, notext_np, 'train')

    return cutted_video


def get_video_text_align_val(frame_dict, tts_dict, asr_dict, moment_info_list, limit_frame_num, notext_np):
    class video:
        def __init__(self):
            self.frame_num = 0
            self.frame_list = []
            self.tts_list = []
            self.asr_list = []
            self.sfx_list = []
            self.moment_start = 1000
            self.moment_end = 0

    # video
    video_dict = dict()
    for k, v in frame_dict.items():
        if k not in video_dict.keys():
            video_dict[k] = video()
        video_dict[k].frame_num = len(v)
        video_dict[k].frame_list = v
    for k, v in tts_dict.items():
        video_dict[k].tts_list = v
    for k, v in asr_dict.items():
        video_dict[k].asr_list = v

    # moment
    for moment_info in moment_info_list:
        k = moment_info[0]
        if int(moment_info[2].split('*')[0]) >= video_dict[k].frame_num:
            continue
        if int(moment_info[2].split('*')[-1]) < 0:
            continue
        moment_start = max(int(moment_info[2].split('*')[0]), 0)
        moment_end = min(int(moment_info[2].split('*')[-1]), video_dict[k].frame_num - 1)
        video_dict[k].sfx_list.append((moment_info[1], moment_start, moment_end))
        video_dict[k].moment_start = min(moment_start, video_dict[k].moment_start)
        video_dict[k].moment_end = max(moment_end, video_dict[k].moment_end)

    # cut video
    cutted_video_dict = dict()
    for k, v in video_dict.items():
        if v.frame_num == 0:
            continue
        if len(v.sfx_list) == 0:
            continue
        if v.frame_num <= limit_frame_num:
            new_k = k + '+' + str(0)
            cut_video_start = 0
            cut_video_end = v.frame_num
            cutted_clip = get_videocut_from_startend(v, cut_video_start, cut_video_end, notext_np, 'val')
            if cutted_clip != None:
                cutted_video_dict[new_k] = cutted_clip

        else:
            cut_num = int(v.frame_num / limit_frame_num)
            cut_reminant = v.frame_num % limit_frame_num
            for i in range(cut_num):
                new_k = k + '+' + str(i)
                cut_video_start = int(i * limit_frame_num)
                cut_video_end = int((i + 1) * limit_frame_num)
                cutted_clip = get_videocut_from_startend(v, cut_video_start, cut_video_end, notext_np, 'val')
                if cutted_clip != None:
                    cutted_video_dict[new_k] = cutted_clip
            if cut_reminant > 0:
                new_k = k + '+' + str(cut_num)
                cut_video_start = int(cut_num * limit_frame_num)
                cut_video_end = v.frame_num
                cutted_clip = get_videocut_from_startend(v, cut_video_start, cut_video_end, notext_np, 'val')
                if cutted_clip != None:
                    cutted_video_dict[new_k] = cutted_clip

    return cutted_video_dict

