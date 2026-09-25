"""
This file contains specific functions for computing losses on the da_heads
file
"""

import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.poolers import Pooler

def binary_classification_accuracy(logits, labels, threshold=0.5):
    logits=logits.view(-1)
    labels=labels.view(-1)
    probabilities = torch.sigmoid(logits)
    predicted_classes = (probabilities >= threshold).float() 
    correct_predictions = (predicted_classes.squeeze() == labels).sum().item()  
    accuracy = correct_predictions / labels.size(0)  
    return accuracy


class DALossComputation(object):
    """
    This class computes the DA loss.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        self.pooler = pooler
        self.avgpool = nn.AvgPool2d(kernel_size=resolution, stride=resolution)

        self.dis_img_result= torch.zeros((2,5),requires_grad=False).cuda()
        self.dis_ins_result= torch.zeros((2,4),requires_grad=False).cuda()

    def prepare_masks(self, targets):
        masks = []
        for targets_per_image in targets:
            is_source = targets_per_image.get_field('is_source')
            mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
            masks.append(mask_per_image)
        return masks

    def __call__(self,  proposals, da_img, da_ins_local,da_ins_labels, targets, levels):

        masks = self.prepare_masks(targets)
        masks = torch.cat(masks, dim=0)


        # img level local loss 
        da_img_loss=0
        for scale_level,da_img_per_level in enumerate(da_img):
            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[masks.bool(), :,:,:] = 1

            self.dis_img_result[0][scale_level]=F.binary_cross_entropy_with_logits(da_img_per_level, da_img_label_per_level).detach()
            self.dis_img_result[1][scale_level]=binary_classification_accuracy(da_img_per_level, da_img_label_per_level)

            da_img_loss+=F.binary_cross_entropy_with_logits(da_img_per_level, da_img_label_per_level)
        da_img_loss/=len(da_img)

        # ins level weight 
        for level_scale in range(4):
            idx_in_level = torch.nonzero(levels == level_scale).squeeze(1)
            if len(idx_in_level)>0:
    
                self.dis_ins_result[0][level_scale]=F.binary_cross_entropy_with_logits(torch.squeeze(da_ins_local)[idx_in_level], da_ins_labels.type(torch.cuda.FloatTensor)[idx_in_level]).detach()
                self.dis_ins_result[1][level_scale]=binary_classification_accuracy(torch.squeeze(da_ins_local)[idx_in_level], da_ins_labels.type(torch.cuda.FloatTensor)[idx_in_level])
               

        da_ins_loss = F.binary_cross_entropy_with_logits(
                            torch.squeeze(da_ins_local), 
                            da_ins_labels.type(torch.cuda.FloatTensor))

        return da_img_loss, da_ins_loss

def make_da_heads_loss_evaluator(cfg):
    loss_evaluator = DALossComputation(cfg)
    return loss_evaluator
