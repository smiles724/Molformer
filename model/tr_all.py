import copy
import torch.nn as nn

from model.tr_spe import Embeddings, FeedForward, Generator3D
from model.tr_afps import MultiRelationEncoder, EncoderLayer
from model.tr_msa import MultiScaleMultiHeadedAttention


def build_model(vocab, tgt, dist_bar, k, N=6, embed_dim=512, ffn_dim=2048, head=8, dropout=0.1):
    assert k > 1
    c = copy.deepcopy
    attn = MultiScaleMultiHeadedAttention(head, embed_dim, dist_bar)
    ff = FeedForward(embed_dim, ffn_dim, dropout)

    model = FullTransformer3D(MultiRelationEncoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N, k),
                              Embeddings(embed_dim, vocab), Generator3D(embed_dim, tgt, dropout))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class FullTransformer3D(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(FullTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, src_mask, dist):
        out = self.encoder(self.src_embed(src), dist, src_mask)
        return self.generator(out)
