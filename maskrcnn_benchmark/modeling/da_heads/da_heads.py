# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved.

from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.layers import GradientScalarLayer
from maskrcnn_benchmark.modeling.poolers import LevelMapper

from .loss import make_da_heads_loss_evaluator


class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head.
    """

    def __init__(self, in_channels):
        super(DAImgHead, self).__init__()

        self.da_img_conv1_layers = []
        self.da_img_conv2_layers = []

        for idx in range(5):
            conv1_block = "da_img_conv1_level{}".format(idx)
            conv2_block = "da_img_conv2_level{}".format(idx)

            conv1_module = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
            conv2_module = nn.Conv2d(512, 1, kernel_size=1, stride=1)

            for module in [conv1_module, conv2_module]:
                nn.init.normal_(module.weight, std=0.002*(idx+1))
                nn.init.constant_(module.bias, 0)

            self.add_module(conv1_block, conv1_module)
            self.add_module(conv2_block, conv2_module)

            self.da_img_conv1_layers.append(conv1_block)
            self.da_img_conv2_layers.append(conv2_block)

    def forward(self, x):
        img_features = []

        for feature, conv1_name, conv2_name in zip(
            x, self.da_img_conv1_layers, self.da_img_conv2_layers
        ):
            feature = F.relu(getattr(self, conv1_name)(feature))
            feature = getattr(self, conv2_name)(feature)
            img_features.append(feature)

        return img_features


class DAInsHead(nn.Module):
    """
    Instance-level Domain Classifier.

    use_dropout=True:
        Used for actual adversarial training.

    use_dropout=False:
        Used for loss/accuracy statistics and adaptive weights.
    """

    def __init__(self, in_channels):
        super(DAInsHead, self).__init__()

        self.da_ins_fc1_layers = []
        self.da_ins_fc2_layers = []
        self.da_ins_fc3_layers = []

        for idx in range(4):
            fc1_block = "da_ins_fc1_level{}".format(idx)
            fc2_block = "da_ins_fc2_level{}".format(idx)
            fc3_block = "da_ins_fc3_level{}".format(idx)

            fc1_module = nn.Linear(in_channels, 1024)
            fc2_module = nn.Linear(1024, 1024)
            fc3_module = nn.Linear(1024, 1)

            for module in [fc1_module, fc2_module, fc3_module]:
                nn.init.normal_(module.weight, std=0.002*(idx+1))
                nn.init.constant_(module.bias, 0)

            self.add_module(fc1_block, fc1_module)
            self.add_module(fc2_block, fc2_module)
            self.add_module(fc3_block, fc3_module)

            self.da_ins_fc1_layers.append(fc1_block)
            self.da_ins_fc2_layers.append(fc2_block)
            self.da_ins_fc3_layers.append(fc3_block)

    def forward(self, x, levels=None, use_dropout=True):
        if levels is None:
            raise ValueError("levels must be provided to DAInsHead.")

        result = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

        fc_layers = zip(
            self.da_ins_fc1_layers,
            self.da_ins_fc2_layers,
            self.da_ins_fc3_layers,
        )

        for level, (fc1_name, fc2_name, fc3_name) in enumerate(fc_layers):
            idx = torch.nonzero(levels == level, as_tuple=False).reshape(-1)

            if idx.numel() == 0:
                continue

            xs = x[idx]

            xs = F.relu(getattr(self, fc1_name)(xs))
            if use_dropout:
                xs = F.dropout(xs, p=0.5, training=self.training)

            xs = F.relu(getattr(self, fc2_name)(xs))
            if use_dropout:
                xs = F.dropout(xs, p=0.5, training=self.training)

            result[idx] = getattr(self, fc3_name)(xs)

        return result


class DomainAdaptationModule(nn.Module):
    """
    Domain Adaptation Component.
    """

    def __init__(self, cfg):
        super(DomainAdaptationModule, self).__init__()

        self.cfg = cfg.clone()
        self.USE_FPN = cfg.MODEL.RPN.USE_FPN

        stage_index = 4
        stage2_relative_factor = 2 ** (stage_index - 1)
        res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS

        if self.USE_FPN:
            num_ins_inputs = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        else:
            num_ins_inputs = res2_out_channels * stage2_relative_factor

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)

        self.consit_weight = cfg.MODEL.DA_HEADS.COS_WEIGHT

        self.grl_img = GradientScalarLayer(
            -cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT
        )
        self.grl_ins = GradientScalarLayer(
            -cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT
        )

        self.grl_img_consist = GradientScalarLayer(
            self.consit_weight * cfg.MODEL.DA_HEADS.DA_IMG_GRL_WEIGHT
        )
        self.grl_ins_consist = GradientScalarLayer(
            self.consit_weight * cfg.MODEL.DA_HEADS.DA_INS_GRL_WEIGHT
        )

        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS

        self.imghead = DAImgHead(in_channels)
        self.inshead = DAInsHead(num_ins_inputs)
        self.loss_evaluator = make_da_heads_loss_evaluator(cfg)

        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES

        lvl_min = -torch.log2(
            torch.tensor(scales[0], dtype=torch.float32)
        ).item()

        lvl_max = -torch.log2(
            torch.tensor(scales[-1], dtype=torch.float32)
        ).item()

        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def forward(
        self,
        proposals,
        img_features,
        da_ins_feature,
        da_ins_labels,
        da_proposals,
        targets=None,
    ):
        if not self.USE_FPN:
            da_ins_feature = self.avgpool(da_ins_feature)

        da_ins_feature = da_ins_feature.view(da_ins_feature.size(0), -1)

        img_grl_fea = [self.grl_img(fea) for fea in img_features]
        ins_grl_fea = self.grl_ins(da_ins_feature)

        levels = self.map_levels(da_proposals).to(ins_grl_fea.device)

        # Training branch: dropout ON.
        da_ins_features = self.inshead(
            ins_grl_fea, levels, use_dropout=True
        )

        # Statistics branch: dropout OFF.
        with torch.no_grad():
            da_ins_features_stat = self.inshead(
                ins_grl_fea.detach(), levels, use_dropout=False
            )

        da_img_features = self.imghead(img_grl_fea)

        if self.training:
            da_img_loss, da_ins_loss = self.loss_evaluator(
                da_proposals,
                da_img_features,
                da_ins_features,
                da_ins_labels,
                targets,
                levels,
                da_ins_stat=da_ins_features_stat,
            )

            weight = self.cfg.MODEL.DA_HEADS.LOSS_WEIGHT

            return {
                "loss_da_image": da_img_loss * weight,
                "loss_da_instance": da_ins_loss * weight,
            }

        return {}


def build_da_heads(cfg):
    if cfg.MODEL.DOMAIN_ADAPTATION_ON:
        return DomainAdaptationModule(cfg)

    return []