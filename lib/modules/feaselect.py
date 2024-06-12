"""
Implmentation of Feature Aggregation for itmAFA.
"""
# coding=utf-8
from email import encoders
from json import encoder
import torch
import torch.nn as nn
import math, random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Attention(nn.Module):
    """
    Attention module implementation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class FeaSelect(nn.Module):
    """
    feature selection module implementation
    """
    def __init__(self, d_pe, d_hidden, embed_size):
        super(FeaSelect, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden
        self.embed_size = embed_size

        #self.attn = Attention(self.embed_size, num_heads=8, qkv_bias=False, attn_drop=0.2, proj_drop=0.2)
        #self.attn = nn.MultiheadAttention(self.embed_size, num_heads=8)
        #self.norm1 = nn.LayerNorm(self.embed_size)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max()) 
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1) # mask: [B, B, max_len, 1]
        return mask

    def forward(self, features, lengths):
        
        mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0] # [0]: select the tensors only
        sorted_features = sorted_features.masked_fill(mask == 0, 0)
        
        # dimensional-wise max selection
        pooled_features = sorted_features[:,0,:]

        pool_weights = None # not be used in current version

        return pooled_features, pool_weights


