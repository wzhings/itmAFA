"""
Loss implementations for itmAFA
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


def get_sim(images, captions):
    similarities = images.mm(captions.t())
    return similarities



class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = get_sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()




class ContrastiveLoss_mixup(nn.Module):
    """
    Contrastive loss with mixup synthetic negative data (max-margin based)
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss_mixup, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')


    def forward(self, im, s):
        scores = get_sim(im, s) 
        # print(f"**** image shape: {im.shape} and text shape: {s.shape}****. ")
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # mixup
        alpha = 1.0
        lam_coef_1 =  np.random.beta(alpha, alpha)
        lam_coef_2 =  np.random.beta(alpha, alpha)

        # generate random permutaiton of texts
        im_mix = lam_coef_1 * im + (1 - lam_coef_1) * s
        s_mix = lam_coef_2 * im + (1 - lam_coef_2) * s

        scores_mix = get_sim(im_mix, s_mix)
        diagonal_mix = scores_mix.diag().view(im_mix.size(0), 1)
        d1_mix = diagonal.expand_as(scores_mix)
        d2_mix = diagonal.t().expand_as(scores_mix)

        # replace the diagonal values
        scores[range(scores.size(0)), range(scores.size(1))] = scores_mix.diag()


        cost_s_mix = ((self.margin/2) + scores - d1_mix).clamp(min=0)
        cost_s_org = (self.margin + scores - d1).clamp(min=0) #original 
        

        cost_im_mix = ((self.margin/2) + scores - d2_mix).clamp(min=0)
        cost_im_org = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()

        cost_s_mix = cost_s_mix.masked_fill_(I, 0)
        cost_s_org = cost_s_org.masked_fill_(I, 0)

        cost_im_mix = cost_im_mix.masked_fill_(I, 0)
        cost_im_org = cost_im_org.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s_mix = cost_s_mix.max(1)[0]
            cost_s_org = cost_s_org.max(1)[0]

            cost_im_mix = cost_im_mix.max(0)[0]
            cost_im_org = cost_im_org.max(0)[0]

        return cost_s_mix.sum() + cost_im_mix.sum() + cost_s_org.sum() + cost_im_org.sum() 



