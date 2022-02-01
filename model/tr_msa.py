import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tr_spe import Embeddings, FeedForward, clones, Generator3D, Feat_Embedding
from model.tr_cpe import Encoder, EncoderLayer


def build_model(vocab, tgt, dist_bar, N=6, embed_dim=512, ffn_dim=2048, head=8, dropout=0.1, extra_feat=None):
    c = copy.deepcopy
    attn = MultiScaleMultiHeadedAttention(head, embed_dim, dist_bar)
    ff = FeedForward(embed_dim, ffn_dim, dropout)
    if extra_feat: extra_feat = Feat_Embedding(extra_feat, embed_dim)

    model = MultiScaleTransformer3D(Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N),
                                    Embeddings(embed_dim, vocab), Generator3D(embed_dim, tgt, dropout),
                                    extra_feat)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class MultiScaleTransformer3D(nn.Module):
    def __init__(self, encoder, src_embed, generator, feat_embed):
        super(MultiScaleTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.feat_embed = feat_embed
        self.generator = generator

    def forward(self, src, src_mask, dist):
        x = self.src_embed(src)
        if self.feat_embed: x += self.feat_embed(src)
        return self.generator(self.encoder(x, dist, src_mask)[:, 0])


#######################################
## attention部分
#######################################

def attention(query, key, value, dist_conv, dist, dist_bar, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    scores *= dist_conv

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    # 多尺度注意力mask
    out = []
    for i in dist_bar:

        # 所有点与中心点始终允许交互
        dist_mask = dist < i
        dist_mask[:, :, 0, :] = 1
        dist_mask[:, :, :, 0] = 1

        # 根据距离mask掉部分attention score
        scores_dist = scores.masked_fill(dist_mask == 0, -1e10)

        p_attn = F.softmax(scores_dist, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        out.append(torch.matmul(p_attn, value))
    return out, p_attn


class MultiScaleMultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dist_bar, dropout=0.1):
        super(MultiScaleMultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        # 4个线性层
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)

        # 处理距离矩阵的1 * 1卷积
        self.cnn = nn.Sequential(nn.Conv2d(1, h, kernel_size=1), nn.ReLU(), nn.Conv2d(h, h, kernel_size=1))

        # 转换不同尺度cat后的向量维度
        self.scale_linear = nn.Sequential(nn.Linear((len(dist_bar) + 1) * embed_dim, embed_dim), nn.ReLU(),
                                          nn.Dropout(p=dropout), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.dist_bar = dist_bar + [1e10]
        self.d_k = embed_dim // h
        self.h = h
        self.attn = None

    def forward(self, query, key, value, dist, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from (B, embed_dim) => (B, head, N, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # dist维度从(B,N,N)扩展为(B,1,N,N)后输入CNN中，得到dist_conv
        dist = dist.unsqueeze(1)
        dist_conv = self.cnn(dist)

        # 2) Apply attention on all the projected vectors in batch.
        x_list, self.attn = attention(query, key, value, dist_conv, dist,
                                      self.dist_bar, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x_list = [x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) for x in x_list]

        # 4) concat不同尺度的attention向量，维度为(B, N, (len(dist_bar) + 1) * d_k)
        x = torch.cat([self.linears[-1](x) for x in x_list], dim=-1)
        return self.scale_linear(x)

