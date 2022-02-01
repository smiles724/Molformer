import copy
import torch
import torch.nn as nn

import sys

sys.path.append("..")
from model.tr_spe import Embeddings, FeedForward, LayerNorm, SublayerConnection, clones, Generator3D
from model.tr_cpe import MultiHeadedAttention


def build_model(vocab, tgt, k, N=6, embed_dim=512, ffn_dim=2048, head=8, dropout=0.1):
    # make sure AFPS exists
    assert k > 1

    c = copy.deepcopy
    attn = MultiHeadedAttention(head, embed_dim)
    ff = FeedForward(embed_dim, ffn_dim, dropout)
    model = MultiRelationTransformer3D(MultiRelationEncoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N, k),
                                       Embeddings(embed_dim, vocab), Generator3D(embed_dim, tgt, dropout))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class MultiRelationTransformer3D(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(MultiRelationTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, src_mask, dist):
        return self.generator(self.encoder(self.src_embed(src), dist, src_mask))


class MultiRelationEncoder(nn.Module):
    def __init__(self, layer, N, k):
        super(MultiRelationEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.k = k

    def forward(self, x, dist, mask):
        for layer in self.layers:
            x, scores = layer(x, dist, mask)

        # go through every sample in the batch
        x_batch, dist_batch, mask_batch = [], [], []
        for i in range(len(dist)):

            # if the molecules have less number of atoms than k, keep them all
            if torch.sum(mask[i]) <= self.k:
                x_batch.append(torch.mean(x[i][mask[i, 0]], dim=0))

            else:
                with torch.no_grad():
                    idx = AFPS(scores[i], dist[i], self.k, mask[i, 0])

                # dim=1则为global average pooling（generator对应输入维度为k），dim=0就是简单的pooling（generator对应输入维度为embed_dim）
                x_batch.append(torch.mean(x[i, idx], dim=0))
        return self.norm(torch.stack(x_batch, dim=0))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.sublayer = SublayerConnection(size, dropout)
        self.size = size

    def forward(self, x, dist, mask):
        att_out = self.self_attn(self.norm(x), self.norm(x), self.norm(x), dist, mask)
        scores = self.self_attn.attn
        x = x + self.dropout(att_out)
        return self.sublayer(x, self.feed_forward), scores


#######################################
## downsampling part
#######################################

def AFPS(scores, dist, k, mask=None):
    # 合并所有的head，遍历head时间消耗大
    scores = torch.mean(scores, dim=0)

    if mask is not None:
        scores = scores[mask][:, mask]
        dist = dist[mask][:, mask]

    # initialize the first point
    scores = torch.sum(scores, dim=-2)

    # 初始化候选点和剩余点，从中心点开始
    remaining_points = [i for i in range(len(dist))]
    solution_set = [remaining_points.pop(0)]

    # incorporate the distance information
    dist = dist / torch.max(dist) + scores.unsqueeze(-1) / torch.max(scores) * 0.1

    while len(solution_set) < k:
        # 得到候选点和剩余点的距离矩阵
        distances = dist[remaining_points][:, solution_set]

        # 更新剩余点的距离，选最大的
        distances = torch.min(distances, dim=-1)[0]
        new_point = torch.argmax(distances).item()
        solution_set.append(remaining_points.pop(new_point))

    return solution_set
