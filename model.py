"""
Author: Jingyu Liu
Date: 2024-01-04 15:52:12
LastEditTime: 2024-05-04 18:56:28
Description: 
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
import pdb

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerLayer(nn.Module):
    """
    One Transformer layer in Transformer Block
    """
    def __init__(self, in_dim, head_num, att_dim, ffn_dim, att_dropout, ffn_dropout):
        super(TransformerLayer, self).__init__()
        self.head_num = head_num
        self.att_dim = att_dim
        self.w_q = nn.Linear(in_dim, head_num * att_dim)
        self.w_k = nn.Linear(in_dim, head_num * att_dim)
        self.w_v = nn.Linear(in_dim, head_num * att_dim)
        self.w_o = nn.Linear(head_num * att_dim, in_dim)
        self.ffn_linear1 = nn.Linear(in_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, in_dim)

        self.ln1 = nn.LayerNorm(in_dim)
        self.ln2 = nn.LayerNorm(in_dim)
        self.att_dropout = nn.Dropout(att_dropout)
        self.ffn_1_dropout = nn.Dropout(ffn_dropout)
        self.ffn_2_dropout = nn.Dropout(ffn_dropout)
        self.activation = F.relu

    def forward(self, x, pos, mask=None):
        """
        INPUT
        x: b, n, d
        pos: b, n, d
        mask: b, n, n

        OUTPUT
        x_out: b, n, d
        """
        b, n, d = x.shape

        q = self.w_q(x + pos).view(b, n, self.head_num, self.att_dim).transpose(1, 2)
        k = self.w_k(x + pos).view(b, n, self.head_num, self.att_dim).permute(0, 2, 3, 1)
        v = self.w_v(x).view(b, n, self.head_num, self.att_dim).transpose(1, 2)

        qk_mat = torch.matmul(q, k) / math.sqrt(self.att_dim)
        if mask != None:
            qk_mat = qk_mat + mask.unsqueeze(1)
        att = F.softmax(qk_mat, dim=-1)
        att_out = torch.matmul(att, v).transpose(1, 2).reshape(b, n, self.head_num * self.att_dim)
        att_out = self.w_o(att_out)

        x = x + self.att_dropout(att_out)
        x = self.ln1(x)
        x_ffn = self.ffn_linear2(self.ffn_1_dropout(self.activation(self.ffn_linear1(x))))
        x_out = x + self.ffn_2_dropout(x_ffn)
        x_out = self.ln2(x_out)

        return x_out


class DetrDecoderLayer(nn.Module):
    """
    One decoder layer in DETR-Decoder Block
    """
    def __init__(self, q_dim, kv_dim, head_num, att_dim, ffn_dim, att_dropout, ffn_dropout):
        super(DetrDecoderLayer, self).__init__()
        self.head_num = head_num
        self.att_dim = att_dim
        # self-att
        self.sa_w_q = nn.Linear(q_dim, head_num * att_dim)
        self.sa_w_k = nn.Linear(q_dim, head_num * att_dim)
        self.sa_w_v = nn.Linear(q_dim, head_num * att_dim)
        self.sa_w_o = nn.Linear(head_num * att_dim, q_dim)
        # cross-att
        self.ca_w_q = nn.Linear(q_dim, head_num * att_dim)
        self.ca_w_k = nn.Linear(kv_dim, head_num * att_dim)
        self.ca_w_v = nn.Linear(kv_dim, head_num * att_dim)
        self.ca_w_o = nn.Linear(head_num * att_dim, q_dim)
        # ffn
        self.ffn_linear1 = nn.Linear(q_dim, ffn_dim)
        self.ffn_linear2 = nn.Linear(ffn_dim, q_dim)
        
        self.satt_dropout = nn.Dropout(att_dropout)
        self.catt_dropout = nn.Dropout(att_dropout)
        self.ffn_1_dropout = nn.Dropout(ffn_dropout)
        self.ffn_2_dropout = nn.Dropout(ffn_dropout)
        self.ln1 = nn.LayerNorm(q_dim)
        self.ln2 = nn.LayerNorm(q_dim)
        self.ln3 = nn.LayerNorm(q_dim)
        self.activation = F.relu

    def forward(self, x1, pos1, x2, pos2, mask):
        """
        INPUT
        x1: b, n1, d1
        pos1: b, n1, d1
        x2: b, n2, d2
        pos2: b, n2, d2
        mask: b, n1, n2

        OUTPUT
        x1_out: b, n1, d1
        """
        b = x1.shape[0]
        n1 = x1.shape[1]
        n2 = x2.shape[1]

        # self-att
        sa_q = self.sa_w_q(x1 + pos1).view(b, n1, self.head_num, self.att_dim).transpose(1, 2)
        sa_k = self.sa_w_k(x1 + pos1).view(b, n1, self.head_num, self.att_dim).permute(0, 2, 3, 1)
        sa_v = self.sa_w_v(x1).view(b, n1, self.head_num, self.att_dim).transpose(1, 2)
        sa_qk_mat = torch.matmul(sa_q, sa_k) / math.sqrt(self.att_dim)
        sa_att = F.softmax(sa_qk_mat, dim=-1)
        sa_att_out = torch.matmul(sa_att, sa_v).transpose(1, 2).reshape(b, n1, self.head_num * self.att_dim)
        sa_att_out = self.sa_w_o(sa_att_out)
        x1 = x1 + self.satt_dropout(sa_att_out)
        x1 = self.ln1(x1)

        # cross-att
        ca_q = self.ca_w_q(x1 + pos1).view(b, n1, self.head_num, self.att_dim).transpose(1, 2)
        ca_k = self.ca_w_k(x2 + pos2).view(b, n2, self.head_num, self.att_dim).permute(0, 2, 3, 1)
        ca_v = self.ca_w_v(x2).view(b, n2, self.head_num, self.att_dim).transpose(1, 2)
        ca_qk_mat = torch.matmul(ca_q, ca_k) / math.sqrt(self.att_dim)
        if mask != None:
            ca_qk_mat = ca_qk_mat + mask.unsqueeze(1)
        ca_att = F.softmax(ca_qk_mat, dim=-1)
        ca_att_out = torch.matmul(ca_att, ca_v).transpose(1, 2).reshape(b, n1, self.head_num * self.att_dim)
        ca_att_out = self.ca_w_o(ca_att_out)
        x1 = x1 + self.catt_dropout(ca_att_out)
        x1 = self.ln2(x1)

        # ffn
        x1_ffn = self.ffn_linear2(self.ffn_1_dropout(self.activation(self.ffn_linear1(x1))))
        x1_out = x1 + self.ffn_2_dropout(x1_ffn)
        x1_out = self.ln3(x1_out)

        return x1_out
    

class DM(nn.Module):
    def __init__(
        self,
        frame_dim,
        text_dim,
        ast_dim,
        hidden_dim,
        encoder_layer_num,
        decoder_layer_num,
        head_num,
        att_dim,
        att_dropout,
        ffn_dropout,
        query_num
        ):
        super().__init__()

        # video
        self.frame_proj = nn.Linear(frame_dim, hidden_dim)
        self.ttsasr_proj = nn.Linear(text_dim, hidden_dim)
        self.tts_lemb = nn.Parameter(torch.randn(1, hidden_dim))
        self.asr_lemb = nn.Parameter(torch.randn(1, hidden_dim))
        self.startend_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transformers = nn.ModuleList([
            TransformerLayer(hidden_dim, head_num, att_dim, int(4 * hidden_dim), att_dropout, ffn_dropout)
            for _ in range(encoder_layer_num)])
        self.query_num = query_num
        self.query_emb = nn.Embedding(query_num, hidden_dim)
        self.detrdecoders = nn.ModuleList([
            DetrDecoderLayer(hidden_dim, hidden_dim, head_num, att_dim, int(4 * hidden_dim), att_dropout, ffn_dropout)
            for _ in range(decoder_layer_num)])
        self.moment_proj = nn.Linear(hidden_dim, hidden_dim)
        self.bbox_proj = MLP(hidden_dim, hidden_dim, 2, 3)

        # SFX
        self.sfx_fusion = nn.Linear(ast_dim + text_dim, hidden_dim)
        self.sfx_lemb = nn.Embedding(6, hidden_dim)
        self.sfx0 = nn.Parameter(torch.randn(hidden_dim))
        self.logit_scale_sfx = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initialize_parameters()

    def initialize_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pe(self, num, hidden_dim, video_emb):
        """
        absolute positional embedding by sinusoidal function
        """
        pe = torch.zeros(num, hidden_dim, dtype=video_emb.dtype, device=video_emb.device)
        position = torch.arange(0, num,
                                dtype=video_emb.dtype,
                                device=video_emb.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2,
                                          dtype=video_emb.dtype,
                                          device=video_emb.device) * -(math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe
    
    def encode_sfx(self, sfx_audio, sfx_text, sfx_tag):
        sfx_tag_emb = self.sfx_lemb(sfx_tag)
        sfx_emb = self.sfx_fusion(torch.cat((sfx_audio, sfx_text), dim=-1)) + sfx_tag_emb
        sfx_emb = sfx_emb / sfx_emb.norm(dim=-1, keepdim=True)

        return sfx_emb

    def encode_video(self, frame_batch, tts_batch, asr_batch,
                     frame_mask, tts_mask, asr_mask,
                     tts_start_pos, tts_end_pos, asr_start_pos, asr_end_pos):
        """
        INPUT
        frame_batch: b, frame_num, d
        tts_batch: b, tts_num, d
        asr_batch: b, asr_num, d
        frame_mask: b, frame_num
        tts_mask: b, tts_num
        asr_mask: b, asr_num
        tts_start_pos: b, tts_num
        tts_end_pos: b, tts_num
        asr_start_pos: b, asr_num
        asr_end_pos: b, asr_num

        OUTPUT
        video_emb: b, frame_tts_asr_num, d
        video_pe: b, frame_tts_asr_num, d
        """
        # emb
        frame_emb = self.frame_proj(frame_batch)  # b, fr_num, d
        tts_emb = self.ttsasr_proj(tts_batch)  # b, tts_num, d
        asr_emb = self.ttsasr_proj(asr_batch)  # b, asr_num, d
        video_emb = torch.cat((frame_emb, tts_emb, asr_emb), dim=1)  # b, frame_tts_asr_num, d

        # pe
        b, frame_num, d = frame_emb.shape
        tts_num = tts_batch.shape[1]
        asr_num = asr_batch.shape[1]
        origin_pe = self.get_pe(frame_num, d, frame_emb)  # fr_num, d
        frame_pe = origin_pe.unsqueeze(0).repeat(b, 1, 1)   # b, fr_num, d
        tts_start = origin_pe[tts_start_pos]  # b, tts_num, d
        tts_end = origin_pe[tts_end_pos]  # b, tts_num, d
        asr_start = origin_pe[asr_start_pos]  # b, asr_num, d
        asr_end = origin_pe[asr_end_pos]  # b, asr_num, d
        tts_pe = self.startend_proj(torch.cat((tts_start, tts_end), dim=-1)) + \
                 self.tts_lemb.unsqueeze(0).repeat(b, tts_num, 1)   # b, tts_num, d
        asr_pe = self.startend_proj(torch.cat((asr_start, asr_end), dim=-1)) + \
                 self.asr_lemb.unsqueeze(0).repeat(b, asr_num, 1)   # b, asr_num, d
        video_pe = torch.cat((frame_pe, tts_pe, asr_pe), dim=1)   # b, frame_tts_asr_num, d

        # mask: b, frame_tts_asr_num, frame_tts_asr_num
        video_mask = (torch.cat((frame_mask, tts_mask, asr_mask), dim=1).unsqueeze(1).repeat(1, video_emb.shape[1], 1))

        # transformer
        for transformer in self.transformers:
            video_emb = transformer(video_emb, video_pe, video_mask)

        return video_emb, video_pe


    def vdsfx(self, frame_batch, tts_batch, asr_batch,
              frame_mask, tts_mask, asr_mask,
              tts_start_pos, tts_end_pos, asr_start_pos, asr_end_pos):
        """
        INPUT
        frame_batch: b, frame_num, d
        tts_batch: b, tts_num, d
        asr_batch: b, asr_num, d
        frame_mask: b, frame_num
        tts_mask: b, tts_num
        asr_mask: b, asr_num
        tts_start_pos: b, tts_num
        tts_end_pos: b, tts_num
        asr_start_pos: b, asr_num
        asr_end_pos: b, asr_num

        OUTPUT
        moment_emb_list: layer_num * (b, query, d)
        moment_bbox_list: layer_num * (b, query, 2)
        video_emb: b, frame_tts_asr_num, d
        """
        # transformer
        video_emb, video_pe = self.encode_video(frame_batch, tts_batch, asr_batch,
                                                frame_mask, tts_mask, asr_mask,
                                                tts_start_pos, tts_end_pos, asr_start_pos, asr_end_pos)

        # detr input
        b = video_emb.shape[0]
        d = video_emb.shape[-1]
        query = torch.zeros(b, self.query_num, d, dtype=video_emb.dtype, device=video_emb.device)
        query_pe = self.query_emb.weight.unsqueeze(0).repeat(b, 1, 1)  # b, query_num, d
        detr_mask = (torch.cat((frame_mask, tts_mask, asr_mask), dim=1).unsqueeze(1)
                     .repeat(1, self.query_num, 1))   # b, query_num, frame_tts_asr_num

        # detr decoder
        moment_emb_list = []
        moment_bbox_list = []
        for detrdecoder in self.detrdecoders:
            query = detrdecoder(query, query_pe, video_emb, video_pe, detr_mask)
            # emb
            moment_emb = self.moment_proj(query)
            moment_emb = moment_emb / moment_emb.norm(dim=-1, keepdim=True)
            moment_emb_list.append(moment_emb)
            # bbox 
            moment_bbox = self.bbox_proj(query).sigmoid()
            moment_bbox_list.append(moment_bbox)

        return moment_emb_list, moment_bbox_list, video_emb
