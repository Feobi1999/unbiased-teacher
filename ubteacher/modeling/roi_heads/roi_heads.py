# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from torch import nn
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from ubteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from ..poolers import ROIPooler
# from detectron2.modeling.poolers import ROIPooler
from .box_head import build_box_head

@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsPseudoLab(StandardROIHeads):


    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
            box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
            box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        else:
            raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images

        #[1000,4]
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512
            import pdb

            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )
            # pdb.set_trace()
        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        # del targets

        if (self.training and compute_loss) or compute_val_loss:

            if len(targets)==0:


                losses, _ = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch,
                )
            else:
                losses, _ = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch=branch,targets=targets
                )

            return proposals, losses


        else:
            if "gt" in branch:
            # pred_instances, box_features_save = self._forward_box(
            #     features, proposals, compute_loss, compute_val_loss, branch
            # )

                #gt_pooling
                pred_instances, predictions = self._forward_box_gt_pooling(
                    features, proposals, compute_loss, compute_val_loss, branch
                )
            else:
                # original
                pred_instances, predictions = self._forward_box(
                    features, proposals, compute_loss, compute_val_loss, branch, targets
                )

            return pred_instances , predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,

        branch: str = "",
        targets: list = None,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:

        features = [features[f] for f in self.box_in_features]



        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals],branch)


        box_features = self.box_head(box_features)   #test [1000 1024]   train [1024,1024]
        predictions = self.box_predictor(box_features)    #train  [1024,9]&&[1024,32]


        # del box_features


        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, proposals)


            if self.train_on_pred_boxes:
                with torch.no_grad():

                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:

            pred_instances, filter_inds = self.box_predictor.inference(predictions, proposals ,branch)
            box_features = box_features[filter_inds]
            # return pred_instances, predictions
            return pred_instances, box_features

    def _forward_box_gt_pooling(
        self,
        features: Dict[str, torch.Tensor],
        gt_instances: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        import pdb

        # pdb.set_trace()
        # box_features = self.box_pooler(features, [x.proposal_boxes for x in gt_instances],branch)
        box_features = self.box_pooler(features, [x.gt_boxes for x in gt_instances],branch)
        box_features = self.box_head(box_features)   #[1000 1024]
        predictions = self.box_predictor(box_features)
        #predictions [0]   [1000,9]
        # [1]    [1000,32]
        # pdb.set_trace()
        # del box_featur es

        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            losses = self.box_predictor.losses(predictions, gt_instances)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, gt_instances
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        gt_instances, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            # proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
            # pdb.set_trace()

            # gt_instances=[p.set('proposal_boxes',p.gt_boxes) for p in gt_instances]
            # print("gt_instances",len(gt_instances))

            pred_instances, filter_inds = self.box_predictor.inference(predictions, gt_instances, branch)

            box_features=box_features[filter_inds]
            import pdb
            # pdb.set_trace()
            # return pred_instances, predictions
            return pred_instances, box_features




    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = ""
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar(
            "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
        )
        storage.put_scalar(
            "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
        )

        return proposals_with_gt
