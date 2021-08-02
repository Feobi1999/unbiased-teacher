"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F
import copy
import pdb
# from pod.plugins.da_faster.models.heads.gcn.models import GCN
# from pod.plugins.da_faster.models.heads.gcn.utils import get_adj

lower_margin = 0.5
margin = 1
class ins_DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        # resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        # sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )

        # self.pooler = pooler
        # self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)
        


    def __call__(self, da_ins, da_ins_labels):
        """
        Arguments:
            da_img (list[Tensor])
            da_img_consist (list[Tensor])
            da_ins (Tensor)
            da_ins_consist (Tensor)
            da_ins_labels (Tensor)
            targets (list[BoxList])

        Returns:
            da_ins_loss (Tensor)
        """

        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), torch.squeeze(da_ins_labels)
        )
        
        return da_ins_loss


class img_DALossComputation(object):
    """
    This class computes the DA loss.
    """
    #
    def __init__(self,cfg):
        self.cfg = cfg.copy()
     
    def prepare_masks(self, targets):
        masks = []
        # targets_tensor=[]
        #source=1 target=0
        for t in targets:
            is_source=torch.tensor(t)
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source==1 else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        # pdb.set_trace()
        return masks

    def __call__(self, da_img, targets, fpn):
        """
        Arguments:
            da_img (list[Tensor])   no fpn#2,1,38,76]
            targets (list[BoxList])

        Returns:
            da_img_loss (Tensor)
        """

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)
        masks=masks.bool()
        # da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        # pdb.set_trace()
        # da_img_flattened = []
        # da_img_labels_flattened = []
        da_img_loss=0
        num_layers= len(da_img)

        for da_img_per_level in da_img:
            N, A, H, W = da_img_per_level.shape
            # pdb.set_trace()
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)  #([2, 38,76,1])
            da_img_label_per_level[masks, :] = 1
            da_img_per_level = da_img_per_level.reshape(N, -1)    #2,2888
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            da_img_loss_per_level = F.binary_cross_entropy_with_logits(
                da_img_per_level, da_img_label_per_level)

            da_img_loss += da_img_loss_per_level
            # fpn=1
        if fpn:
            da_img_loss/=num_layers
            # pdb.set_trace()
        return da_img_loss


def make_ins_heads_loss_evaluator(cfg):
    ins_loss_evaluator = ins_DALossComputation(cfg)
    return ins_loss_evaluator

def make_img_heads_loss_evaluator(cfg):
    img_loss_evaluator = img_DALossComputation(cfg)
    return img_loss_evaluator






def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):
    """
    Consistency regularization as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    """
    loss = []
    len_ins = ins_fea.size(0)
    intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
    for img_fea_per_level in img_feas:
        N, A, H, W = img_fea_per_level.shape
        img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
        img_feas_per_level = []
        assert N==2, \
            "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
        for i in range(N):
            img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
            img_feas_per_level.append(img_fea_mean)
        img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
        loss_per_level = torch.abs(img_feas_per_level - ins_fea)
        loss.append(loss_per_level)
    loss = torch.cat(loss, dim=1)
    if size_average:
        return loss.mean()
    return loss.sum()


