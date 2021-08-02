# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from utils import FDA_source_to_target
import torch
import numpy as np
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, target_batched_inputs=None, branch="supervised", domain_stats=None, given_proposals=None, val_mode=False
    ):


        '''
        batched_inputs: list of inputs
        len(batched_inputs) = image_per_batch
        batched_inputs[0] dict keys() ['file_name', 'height', 'width', 'image_id', 'image'])
        '''

        #for fft try


        if (not self.training) and (not val_mode):

            if domain_stats == "s_to_t":
                # source_images = [x["image"].to(self.device) for x in batched_inputs]
                # target_images = [x["image"].to(self.device) for x in targte_batched_inputs]

                src_to_tgt_style = batched_inputs.copy()
                tgt_to_src_style = target_batched_inputs.copy()

                import pdb
                pdb.set_trace()
                # if mean_img.shape[-1] < 2:
                batch_size = len(batched_inputs)

                for i in range(0, batch_size):

                    src_image = batched_inputs[i]['image'].unsqueeze(0)
                    tgt_image = target_batched_inputs[i]['image'].unsqueeze(0)

                    H, W = src_image.shape
                    mean_img = IMG_MEAN.repeat(1, 1, H, W)
                    # mean_img2 = IMG_MEAN.repeat(1,1, H2, W2)
                    # import pdb
                    # pdb.set_trace()
                    src_image = src_image.clone() - mean_img
                    tgt_image = tgt_image.clone() - mean_img
                    # src_image.resize()
                    import pdb
                    # pdb.set_trace()
                    try:
                        # print('one')

                        src_in_trg = FDA_source_to_target(src_image, tgt_image, L=0.1)  # src_lbl
                        # trg_img = trg_in_trg.clone() - mean_img
                        trg_in_src = FDA_source_to_target(tgt_image, src_image, L=0.1)
                    except RuntimeError:
                        pdb.set_trace()
                    # pdb.set_trace()
                    # fft_src_in_tgt = src_in_trg.squeeze(0).resize((W,H),Image.BICUBIC)
                    fft_src_in_tgt = src_in_trg.squeeze(0)
                    fft_tgt_in_src = trg_in_src.squeeze(0)

                    src_to_tgt_style[i]['image'] = fft_src_in_tgt
                    tgt_to_src_style[i]['image'] = fft_tgt_in_src


            return self.inference(tgt_to_src_style)




        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
