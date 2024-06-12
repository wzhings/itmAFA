"""
Feature aggregation module implementation for itmAFA
"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.feaselect import FeaSelect
from lib.modules.mlp import MLP, MLPdp

import logging

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def feature_aggr(features, r, lengths):
    """
    feature aggregation module
    """

    t1 = lengths.min().item()
    r = int(min(r, t1//2))
    
    # updates lengths
    lengths = lengths - r

    features1 = l2norm(features, dim=-1)
    src, dst = features1[..., ::2, :], features1[..., 1::2, :]

    n, t1, c = src.shape

    scores = src@dst.transpose(-1,-2)

    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    unm_idx = edge_idx[..., r:, :]  #unmerged
    src_idx = edge_idx[..., :r, :]  #merged
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
    src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
    mode = 'mean'
    dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode) 

    return torch.cat([unm, dst], dim=1), lengths      



def get_text_encoder(embed_size, opt, no_txtnorm=False):
    return EncoderText(embed_size, opt=opt, no_txtnorm=no_txtnorm)


def get_image_encoder(data_name, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """
    a wraper of visual encoder
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm) 
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm) 
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.feaselect = FeaSelect(32, 32, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """
        Extract image feature vectors.
        """
        features = self.fc(images)

        if self.precomp_enc_type == 'basic':
            features = self.mlp(images) + features

        # feature aggregation
        features, image_lengths = feature_aggr(features, r=features.shape[1]//20, lengths=image_lengths)
        # feature selection
        features, pool_weights = self.feaselect(features, image_lengths) 
 
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, opt, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size) 
        self.feaselect = FeaSelect(32, 32, embed_size)

    def forward(self, x, cap_lengths):
        """
        handle the textual information
        """
        bert_attention_mask = (x != 0).float()
        output = self.bert(x, bert_attention_mask)
        bert_emb = output.last_hidden_state 
        bert_pooler = output.pooler_output

        cap_len = cap_lengths
        cap_emb_copy = self.linear(bert_emb) 

        # feature aggregation
        cap_emb, cap_len = feature_aggr(cap_emb_copy, r=cap_emb_copy.shape[1]//20, lengths=cap_len)
        # feature selection
        pooled_features, pool_weights = self.feaselect(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features, cap_emb_copy
