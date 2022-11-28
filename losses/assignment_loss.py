import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


def compute_mask(feat):
    batch_size, fdim, h, w = feat.shape
    norms = torch.norm(feat, p=2, dim=1).view(batch_size, h * w)

    norms -= norms.min(dim=-1, keepdim=True)[0]
    norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
    mask = norms.view(batch_size, 1, h, w)

    return mask.detach()

def feature_similarity(feat_q, feat_k):
    batch_size, fdim, h, w = feat_q.shape
    feat_q = feat_q.view(batch_size, fdim, -1)
    feat_k = feat_k.view(batch_size, fdim, -1)

    feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0,2,1), F.normalize(feat_k, dim=1))
    return feature_sim

class LinearSumAssignment(nn.Module):
    def __init__(self):
        super(LinearSumAssignment, self).__init__()

    def forward(self, feat2d, pos_ind, neg_ids):
        b, c , h, w = feat2d.shape
        p = h * w // 2 # 50 percent of image used for p2p matching
        mask = compute_mask(feat2d)
        maskTr = mask.view(b, 1, -1).clone()
        s, i = torch.sort(maskTr, dim=2)
        bMask = torch.ones_like(maskTr)
        bMask[maskTr < s[:, :, p : p + 1]] = 0
        bMask = bMask.view(mask.shape)

        featC, featI = torch.split(feat2d, b //2 , 0)
        # featPos =
        sim = feature_similarity(featC * bMask[: b // 2], featI * bMask[b // 2 :])
        loss = 0
        for simMat in sim:
            row_ind, col_ind = linear_sum_assignment(simMat.cpu().detach(), True)
            loss += (1 - simMat[row_ind, col_ind].sum()/p )
        return loss / b
