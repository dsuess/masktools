import torch
from typing import Callable


def _compute_maskiou(masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
    assert masks_a.ndim == masks_b.ndim == 3
    masks_a = masks_a.flatten(start_dim=1)
    masks_b = masks_b.flatten(start_dim=1)

    intersection = (masks_a[:, None] * masks_b[None, :]).sum(dim=2)
    area_a = masks_a.sum(dim=1)
    area_b = masks_b.sum(dim=1)
    union = area_a[:, None] + area_b[None, :] - intersection

    return intersection.to(torch.float32) / union


def masknms(
    masks: torch.Tensor,
    confidences: torch.Tensor,
    iou_threshold=0.7,
    _iou_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = _compute_maskiou,
) -> torch.Tensor:
    """Performs NMS on masks.

    Arguments:
        masks {torch.Tensor} -- Tensor of shape `[N, H, W]` where `N` is the number
          of masks and `H`/`W` the mask height/width. The values should be 0 or 1.
        confidences {torch.Tensor} -- Tensor of shape `N` containing the confidence
          values (or the logits; these are only used for sorting)

    Keyword Arguments:
        iou_threshold {float} -- Masks with IOU larger than this value are merged
          (default: {0.7})

    Returns
    """
    order = confidences.argsort(descending=True)
    masks = masks[order]
    keep = []

    for idx, current in enumerate(masks):
        iou = _iou_func(current[None], masks[keep])

        if not torch.any(iou >= iou_threshold):
            keep.append(idx)

    return masks[keep]
