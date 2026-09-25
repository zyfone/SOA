import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.poolers import Pooler


@torch.no_grad()
def binary_classification_accuracy(logits, labels, threshold=0.5):
    logits = logits.reshape(-1)
    labels = labels.reshape(-1).to(device=logits.device)

    if logits.numel() == 0:
        return logits.new_tensor(float("nan"))
    # if logits.numel() == 0:
    #     return logits.new_tensor(0.0)

    predicted_classes = (torch.sigmoid(logits) >= threshold).to(labels.dtype)
    return (predicted_classes == labels).float().mean()


class DALossComputation(object):
    """
    Compute domain-adaptation losses at image level and instance level.
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()

        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO

        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.avgpool = nn.AvgPool2d(
            kernel_size=resolution,
            stride=resolution,
        )

        self.dis_img_result = None
        self.dis_ins_result = None

    def prepare_masks(self, targets, device):
        masks = []

        for targets_per_image in targets:
            is_source = targets_per_image.get_field("is_source")

            if is_source.numel() == 0:
                raise ValueError(
                    "Empty is_source field: image-level domain label "
                    "cannot be inferred from zero instances."
                )

            if not torch.all(is_source == is_source[0]):
                raise ValueError(
                    "Inconsistent is_source labels found within one image."
                )

            masks.append(
                is_source[0].to(device=device, dtype=torch.bool).reshape(1)
            )

        return torch.cat(masks, dim=0)

    def __call__(
        self,
        proposals,
        da_img,
        da_ins_local,
        da_ins_labels,
        targets,
        levels,
    ):
        if len(da_img) == 0:
            raise ValueError("da_img must contain at least one feature level.")

        img_device = da_img[0].device
        masks = self.prepare_masks(targets, device=img_device)

        if masks.numel() != da_img[0].shape[0]:
            raise ValueError(
                f"targets batch size {masks.numel()} does not match "
                f"da_img batch size {da_img[0].shape[0]}."
            )

        # _, _, height, width = da_img[0].shape

        self.dis_img_result = torch.full(
            (2, len(da_img)),
            float("nan"),
            device=img_device,
            dtype=torch.float32,
        )
        # self.dis_img_result = torch.zeros(
        #     (2, len(da_img)),
        #     device=img_device,
        #     dtype=torch.float32,
        # )

        img_losses = []

        for scale_level, da_img_per_level in enumerate(da_img):
            # da_img_per_level = F.interpolate(
            #     da_img_per_level,
            #     size=(height, width),
            #     mode="bilinear",
            #     align_corners=True,
            # )

            da_img_label_per_level = torch.zeros_like(da_img_per_level)
            da_img_label_per_level[masks] = 1.0

            level_loss = F.binary_cross_entropy_with_logits(
                da_img_per_level,
                da_img_label_per_level,
            )

            img_losses.append(level_loss)

            self.dis_img_result[0, scale_level] = level_loss.detach().float()
            self.dis_img_result[1, scale_level] = (
                binary_classification_accuracy(
                    da_img_per_level,
                    da_img_label_per_level,
                ).float()
            )

        da_img_loss = torch.stack(img_losses).mean()

        da_ins_logits = da_ins_local.reshape(-1)
        da_ins_labels = da_ins_labels.reshape(-1).to(
            device=da_ins_logits.device,
            dtype=da_ins_logits.dtype,
        )
        levels = levels.reshape(-1).to(device=da_ins_logits.device)

        if not (
            da_ins_logits.numel()
            == da_ins_labels.numel()
            == levels.numel()
        ):
            raise ValueError(
                "da_ins_local, da_ins_labels and levels must have "
                "the same number of instances."
            )

        num_instance_levels = 4

        if levels.numel() > 0 and (
            (levels < 0) | (levels >= num_instance_levels)
        ).any():
            raise ValueError("levels must be in the range [0, 3].")

        self.dis_ins_result = torch.full(
            (2, num_instance_levels),
            float("nan"),
            device=da_ins_logits.device,
            dtype=torch.float32,
        )
        # self.dis_ins_result = torch.zeros(
        #     (2, num_instance_levels),
        #     device=da_ins_logits.device,
        #     dtype=torch.float32,
        # )

        for level_scale in range(num_instance_levels):
            idx_in_level = torch.nonzero(
                levels == level_scale,
                as_tuple=False,
            ).reshape(-1)

            if idx_in_level.numel() > 0:
                level_logits = da_ins_logits[idx_in_level]
                level_labels = da_ins_labels[idx_in_level]

                level_loss = F.binary_cross_entropy_with_logits(
                    level_logits,
                    level_labels,
                )

                self.dis_ins_result[0, level_scale] = level_loss.detach().float()
                self.dis_ins_result[1, level_scale] = (
                    binary_classification_accuracy(
                        level_logits,
                        level_labels,
                    ).float()
                )

        if da_ins_logits.numel() == 0:
            da_ins_loss = da_ins_logits.sum() * 0.0
        else:
            da_ins_loss = F.binary_cross_entropy_with_logits(
                da_ins_logits,
                da_ins_labels,
            )

        return da_img_loss, da_ins_loss


def make_da_heads_loss_evaluator(cfg):
    return DALossComputation(cfg)