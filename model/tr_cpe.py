import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("..")
from model.tr_spe import Embeddings, FeedForward
from model.tr_spe import LayerNorm, SublayerConnection, clones, Generator3D


def build_model(vocab, tgt, N=6, embed_dim=512, ffn_dim=2048, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, embed_dim)
    ff = FeedForward(embed_dim, ffn_dim, dropout)
    model = MultiRepresentationTransformer3D(Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N),
                                             Embeddings(embed_dim, vocab), Generator3D(embed_dim, tgt, dropout))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class MultiRepresentationTransformer3D(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(MultiRepresentationTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, src_mask, dist):
        return self.generator(self.encoder(self.src_embed(src), dist, src_mask)[:, 0])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, dist, mask):
        for layer in self.layers:
            x = layer(x, dist, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """MultiRelationEncoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, dist, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist, mask))
        return self.sublayer[1](x, self.feed_forward)


#######################################
## attention part
#######################################

def attention(query, key, value, dist_conv, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    scores *= dist_conv

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        # four linear layers
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)

        # 1 * 1 convolution operator
        self.cnn = nn.Sequential(nn.Conv2d(1, h, kernel_size=1), nn.ReLU(), nn.Conv2d(h, h, kernel_size=1))

        self.dropout = nn.Dropout(p=dropout)
        self.d_k = embed_dim // h
        self.h = h
        self.attn = None

    def forward(self, query, key, value, dist_conv, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from (B, embed_dim) => (B, head, N, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]

        # dist维度从(B, N, N)扩展为(B,1,N,N)后输入CNN中
        dist_conv = self.cnn(dist_conv.unsqueeze(1))

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dist_conv, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
