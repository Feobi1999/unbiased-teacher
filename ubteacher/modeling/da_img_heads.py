# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn
from .gradient_scalar_layer import GradientScalarLayer
from .loss import make_img_heads_loss_evaluator
import pdb
import torch.nn as nn
import torch.nn.functional as F

import copy

__all__ = ['imgDomainAdaptationModule']
@MODULE_ZOO_REGISTRY.register('da_img_head')
class imgDomainAdaptationModule(torch.nn.Module):
    """
    Module for  img Domain classifier  Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """
    def __init__(self, inplanes, cfg):
        super(imgDomainAdaptationModule, self).__init__()

        self.cfg = copy.deepcopy(cfg)
        self.prefix = 'ImgHead'
        self.consist = cfg['da_consist_weight']
        # stage_index = 4
        # stage2_relative_factor = 2 ** (stage_index - 1)
        # res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS        
        # self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        if isinstance(inplanes, list):
            assert len(inplanes) == 1, 'single input is expected, but found:{} '.format(inplanes)
            inplanes = inplanes[0]
        assert isinstance(inplanes, int)
        self.inplanes = inplanes  #256
        self.img_weight = cfg['img_head_weight']
        self.fpn = cfg['fpn']
        self.grl_img = GradientScalarLayer(-1.0*self.cfg['img_grl_weight'])
        self.grl_img_consist = GradientScalarLayer(1.0 * cfg['img_grl_weight'])
        # in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(self.inplanes)
        self.img_loss_evaluator = make_img_heads_loss_evaluator(cfg)



    def forward(self, input):
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

        output = {}
        if self.training:
            losses = self.get_loss(input)
            output.update(losses)
        return output



    def get_loss(self,input):


        '''
        feature no fpn  [2,1024,38,76]
        using FPN [N,C,H,W]
        img_features[0].shape
        torch.Size([2, 256, 38, 76])
        img_features[1].shape
        torch.Size([2, 256, 19, 38])
        img_features[2].shape
        torch.Size([2, 256, 10, 19])
        img_features[3].shape
        torch.Size([2, 256, 5, 10])'''

        img_features = input['features']
        target = input['image_sources']
        # pdb.set_trace()
        img_grl_fea = [self.grl_img(fea) for fea in img_features]  #[2,1024,38,76]
        da_img_features = self.imghead(img_grl_fea)   #[2,1,38,76]
        da_img_loss= self.img_loss_evaluator(
            da_img_features, target, self.fpn
        )
        # pdb.set_trace()



        if self.img_weight > 0:
            img_da_loss = self.img_weight * da_img_loss
            if self.consist > 0:
                img_grl_consist_fea = [self.grl_img_consist(fea) for fea in img_features]  # 2,1024,38,76
                da_img_consist_features = self.imghead(img_grl_consist_fea)  # 2,1,38,76
                da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]  # 2,1,38,76

                return {
                    self.prefix + '.da_loss': img_da_loss,
                    'img_grl_feature': da_img_consist_features }
            else:
                return {
                    self.prefix + '.da_loss': img_da_loss}
        else:
            return {
            }


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
        print("inchannels",in_channels)
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features