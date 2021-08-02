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
import pdb

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
        self.consist_weight=cfg['da_consist_weight']
        self.grl_ins = GradientScalarLayer(-1.0*cfg['ins_grl_weight'])
        self.inshead = DAInsHead(inplanes)
        self.grl_ins_consist = GradientScalarLayer(1.0 * cfg['ins_grl_weight'])
        self.loss_evaluator = make_ins_heads_loss_evaluator(cfg)
        self.consistency_loss=consistency_loss
        # self.resnet_backbone=cfg['res_no_fpn']

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
            da_ins_feature=input['ins_feature']

            # pdb.set_trace()
            # if self.resnet_backbone:
            #     da_ins_feature = self.avgpool(da_ins_feature)
            da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)
            da_ins_labels=input['ins_domain_label']

            ins_grl_consist_fea = self.grl_ins_consist(da_ins_feature)
            ins_grl_fea = self.grl_ins(da_ins_feature)

            da_ins_consist_features = self.inshead(ins_grl_consist_fea)
            da_ins_features = self.inshead(ins_grl_fea)   #[N,1]
            da_ins_consist_features = da_ins_consist_features.sigmoid()
            # da_img_feature = input['img_grl_feature']
            # consistency_loss=self.consistency_loss(da_img_feature,da_ins_consist_features,da_ins_labels,size_average=True)
            da_ins_loss = self.loss_evaluator(
                da_ins_features, da_ins_labels
            )

            if self.ins_weight > 0:
                da_ins_loss = self.ins_weight * da_ins_loss
                if self.consist_weight>0:
                    consistency_loss=self.consist_weight*consistency_loss
                    return {
                        self.prefix + '.da_loss': da_ins_loss,
                        self.prefix+'.consist_loss':consistency_loss}
                else:
                    return {self.prefix + '.da_loss': da_ins_loss}
        return {}



    def gcn_adaptive_loss(self, pooled_feat, cls_prob, rois, tgt_pooled_feat, tgt_cls_prob, tgt_rois, batch_size, epsilon = 1e-6):

        # get the feature embedding of every class for source and target domains wiith GCN
        pooled_feat = pooled_feat.view(batch_size, pooled_feat.size(0) // batch_size, pooled_feat.size(1))
        cls_prob = cls_prob.view(batch_size, cls_prob.size(0) // batch_size, cls_prob.size(1))
        tgt_pooled_feat = tgt_pooled_feat.view(batch_size, tgt_pooled_feat.size(0) // batch_size,
                                               tgt_pooled_feat.size(1))
        tgt_cls_prob = tgt_cls_prob.view(batch_size, tgt_cls_prob.size(0) // batch_size, tgt_cls_prob.size(1))

        num_classes = cls_prob.size(2)
        class_feat = list()
        tgt_class_feat = list()

        for i in range(num_classes):
            tmp_cls_prob = cls_prob[:, :, i].view(cls_prob.size(0), cls_prob.size(1), 1)
            tmp_class_feat = pooled_feat * tmp_cls_prob
            tmp_feat = list()
            tmp_weight = list()

            for j in range(batch_size):
                tmp_batch_feat_ = tmp_class_feat[j, :, :]
                tmp_batch_weight_ = tmp_cls_prob[j, :, :]
                tmp_batch_adj = get_adj(rois[j, :, :])

                # graph-based aggregation
                tmp_batch_feat = torch.mm(tmp_batch_adj, tmp_batch_feat_)
                tmp_batch_weight = torch.mm(tmp_batch_adj, tmp_batch_weight_)

                tmp_feat.append(tmp_batch_feat)
                tmp_weight.append(tmp_batch_weight)

            tmp_class_feat_ = torch.stack(tmp_feat, dim = 0)
            tmp_class_weight = torch.stack(tmp_weight, dim = 0)
            tmp_class_feat = torch.sum(torch.sum(tmp_class_feat_, dim=1), dim = 0) / (torch.sum(tmp_class_weight) + epsilon)
            class_feat.append(tmp_class_feat)

            tmp_tgt_cls_prob = tgt_cls_prob[:, :, i].view(tgt_cls_prob.size(0), tgt_cls_prob.size(1), 1)
            tmp_tgt_class_feat = tgt_pooled_feat * tmp_tgt_cls_prob
            tmp_tgt_feat = list()
            tmp_tgt_weight = list()

            for j in range(batch_size):
                tmp_tgt_batch_feat_ = tmp_tgt_class_feat[j, :, :]
                tmp_tgt_batch_weight_ = tmp_tgt_cls_prob[j, :, :]
                tmp_tgt_batch_adj = get_adj(tgt_rois[j, :, :])

                # graph-based aggregation
                tmp_tgt_batch_feat = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_feat_)
                tmp_tgt_batch_weight = torch.mm(tmp_tgt_batch_adj, tmp_tgt_batch_weight_)

                tmp_tgt_feat.append(tmp_tgt_batch_feat)
                tmp_tgt_weight.append(tmp_tgt_batch_weight)

            tmp_tgt_class_feat_ = torch.stack(tmp_tgt_feat, dim = 0)
            tmp_tgt_class_weight = torch.stack(tmp_tgt_weight, dim = 0)
            tmp_tgt_class_feat = torch.sum(torch.sum(tmp_tgt_class_feat_, dim=1), dim = 0) / (torch.sum(tmp_tgt_class_weight) + epsilon)
            tgt_class_feat.append(tmp_tgt_class_feat)

        class_feat = torch.stack(class_feat, dim = 0)
        tgt_class_feat = torch.stack(tgt_class_feat, dim = 0)
        # get the intra-class and inter-class adaptation loss
        intra_loss = 0
        inter_loss = 0

        for i in range(class_feat.size(0)):
            tmp_src_feat_1 = class_feat[i, :]
            tmp_tgt_feat_1 = tgt_class_feat[i, :]

            intra_loss = intra_loss + self.distance(tmp_src_feat_1, tmp_tgt_feat_1)

            for j in range(i+1, class_feat.size(0)):
                tmp_src_feat_2 = class_feat[j, :]
                tmp_tgt_feat_2 = tgt_class_feat[j, :]

                inter_loss = inter_loss + torch.pow(
                    (self.margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2))) / self.margin,
                    2) * torch.pow(
                    torch.max(self.margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_src_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (self.margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2))) / self.margin,
                    2) * torch.pow(
                    torch.max(self.margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_tgt_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (self.margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2))) / self.margin,
                    2) * torch.pow(
                    torch.max(self.margin - torch.sqrt(self.distance(tmp_src_feat_1, tmp_tgt_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

                inter_loss = inter_loss + torch.pow(
                    (self.margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2))) / self.margin,
                    2) * torch.pow(
                    torch.max(self.margin - torch.sqrt(self.distance(tmp_tgt_feat_1, tmp_src_feat_2)),
                              torch.tensor(0).float().cuda()), 2.0)

        intra_loss = intra_loss / class_feat.size(0)
        inter_loss = inter_loss / (class_feat.size(0) * (class_feat.size(0) - 1) * 2)

        return intra_loss, inter_loss