# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function

import torch
import torch.nn.functional as F
from ..gradient_scalar_layer import GradientScalarLayer
from torch import nn

# from .loss import make_da_heads_loss_evaluator
from ..loss import make_img_heads_loss_evaluator

class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()

        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features


class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()
        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x


class ImgDomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self,cfg):
        super(ImgDomainAdaptationModule, self).__init__()

        self.grl_img = GradientScalarLayer(-1.0 * 0.1)
        # self.grl_img = GradientScalarLayer(-1.0 * self.cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT)
        in_channels = 256
        self.imghead = DAImgHead(in_channels)
        self.img_loss_evaluator = make_img_heads_loss_evaluator(cfg)
        self.img_weight = 1


    def forward(self, img_features, targets=None):
        """
        Arguments:
            img_features (list[Tensor]): features computed from the images that are
                used for computing the predictions.
            da_ins_feature (Tensor): instance-level feature vectors
            da_ins_labels (Tensor): domain labels for instance-level feature vectors
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """

        import pdb
        # pdb.set_trace()
        # feature_name = ["p4", "p5"]
        # img_features = [img_features[f] for f in feature_name]
        # img_features = [img_features[fea] for fea in img_features]
        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        da_img_features = self.imghead(img_grl_fea)


        if self.training:
            da_img_loss = self.img_loss_evaluator(
                da_img_features,
                targets ,fpn=True
            )
            losses = {}
            if self.img_weight > 0:
                losses["loss_da_image"] = self.img_weight * da_img_loss
            return losses
        return {}


def build_img_da_heads():
    # if cfg.MODEL.DOMAIN_ADAPTATION_ON:
    return ImgDomainAdaptationModule
# return []
