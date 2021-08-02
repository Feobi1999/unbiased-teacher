# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from __future__ import print_function
import torch
import copy
import torch.nn.functional as F
from torch import nn
from .gradient_scalar_layer import GradientScalarLayer
from pod.utils.registry_factory import MODULE_ZOO_REGISTRY
from .loss import make_ins_heads_loss_evaluator
from .loss import consistency_loss
from.loss import gcn_adaptive_loss
import pdb
from pod.plugins.da_faster.models.heads.gcn.models import GCN
from pod.plugins.da_faster.models.heads.gcn.utils import get_adj

__all__ = ['insDomainAdaptationModule']
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


@MODULE_ZOO_REGISTRY.register('da_ins_head')

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
class insDomainAdaptationModule(torch.nn.Module):
    """
    Module for Domain Adaptation Component. Takes feature maps from the backbone and instance
    feature vectors, domain labels and proposals. Works for both FPN and non-FPN.
    """

    def __init__(self,inplanes, cfg):
        super(insDomainAdaptationModule, self).__init__()

        self.cfg = copy.deepcopy(cfg)
        self.prefix = 'InsHead'
        # stage_index = 4
        # stage2_relative_factor = 2 ** (stage_index - 1)
        # res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
        # num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM if cfg.MODEL.BACKBONE.CONV_BODY.startswith('V') else res2_out_channels * stage2_relative_factor
        
        # self.resnet_backbone = cfg.MODEL.BACKBONE.CONV_BODY.startswith('R')
        # self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.ins_weight = cfg['ins_head_weight']
        if self.ins_weight>0:
            self.consist_weight=cfg['da_consist_weight']
            self.grl_ins = GradientScalarLayer(-1.0*cfg['ins_grl_weight'])
            self.inshead = DAInsHead(inplanes)
            self.grl_ins_consist = GradientScalarLayer(1.0 * cfg['ins_grl_weight'])
            self.loss_evaluator = make_ins_heads_loss_evaluator(cfg)
            self.consistency_loss=consistency_loss
        self.domain_on = cfg['domain_on']
        if self.domain_on:
            self.RCNN_adapt_feat = nn.Linear(inplanes, 64)
            nn.init.normal_(self.RCNN_adapt_feat.weight, std=0.01)
            nn.init.constant_(self.RCNN_adapt_feat.bias, 0)

        # self.resnet_backbone=cfg['res_no_fpn']
        self.gcn_adaptive_loss=gcn_adaptive_loss

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
        if self.training:
            batch_size=1
            pooled_feature=input['ins_feature']    #1024 2048
            source_feature=input['source_ins_feature']   #256 2048
            target_feature=input['target_ins_feature']   #256 2048
            # rois=input['dt_bboxes']
            # print(rois.shape)   #2097 7
            source_cls_prob = input['source_cls_pred']  #256 2
            tgt_cls_prob = input['target_cls_pred']  #256 2
            source_rois=input['source_roi']
            tgt_rois=input['target_roi']
            num_proposal=source_rois.size(0)
            num_proposal2=tgt_rois.size(0)
            # if self.resnet_backbone:
            #     da_ins_feature = self.avgpool(da_ins_feature)

            # pdb.set_trace()
            if self.ins_weight > 0:
                da_ins_feature = pooled_feature.view(pooled_feature.size(0), -1)
                da_ins_labels=input['ins_domain_label']

                ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
                ins_grl_fea = self.grl_ins(da_ins_feature)

                da_ins_consist_features = self.inshead(ins_grl_consist_fea)
                da_ins_features = self.inshead(ins_grl_fea)   #[N,1]
                da_ins_loss = self.loss_evaluator(
                    da_ins_features, da_ins_labels
                )
                da_ins_loss = self.ins_weight * da_ins_loss
                if self.consist_weight>0:
                    da_ins_consist_features = da_ins_consist_features.sigmoid()
                    da_img_feature = input['img_grl_feature']

                    consistency_loss = self.consistency_loss(da_img_feature, da_ins_consist_features, da_ins_labels,
                                                             size_average=True)
                    consistency_loss=self.consist_weight*consistency_loss
                    return {
                        self.prefix + '.da_loss': da_ins_loss,
                        self.prefix+'.consist_loss':consistency_loss}
                else:
                    return {self.prefix + '.da_loss': da_ins_loss}

            else:
                if self.domain_on:

                    source_output = source_rois.new(batch_size, num_proposal, 5).zero_()
                    source_output[0]=source_rois
                    # source_output[0, :num_proposal, 1:] = source_rois
                    target_output = tgt_rois.new(batch_size, num_proposal2, 5).zero_()
                    target_output[0] =tgt_rois
                    source_feature=self.RCNN_adapt_feat(source_feature)
                    target_feature=self.RCNN_adapt_feat(target_feature)
                    RCNN_loss_intra, RCNN_loss_inter = self.gcn_adaptive_loss(source_feature, source_cls_prob, source_output,target_feature,
                                                                              tgt_cls_prob, target_output, batch_size)

                    # pdb.set_trace()
                    return {self.prefix+'.intra_loss':RCNN_loss_intra,
                            self.prefix + '.inter_loss': RCNN_loss_inter,
                    }
        else:
            return {}