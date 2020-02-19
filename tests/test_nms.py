import numpy as np
import torch
from PIL import Image, ImageDraw
from typing import List, Tuple, NamedTuple
import pytest as pt
from masktools import pure_torch


def create_circle_pic(
    centers: List[Tuple[int, int]],
    radii: List[int],
    imgsize: Tuple[int, int] = (200, 200),
):
    assert len(centers) == len(radii)
    image = Image.new("1", imgsize)
    draw = ImageDraw.Draw(image)

    for (x, y), r in zip(centers, radii):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=1)

    return torch.from_numpy(np.array(image))


def create_masks(
    centers: List[Tuple[int, int]],
    radii: List[int],
    imgsize: Tuple[int, int] = (200, 200),
):
    assert len(centers) == len(radii)
    return torch.stack(
        [
            create_circle_pic([center], [radius], imgsize=imgsize)
            for center, radius in zip(centers, radii)
        ]
    )


class Case(NamedTuple):
    masks_in: torch.Tensor
    scores: torch.Tensor
    masks_out: torch.Tensor
    iou_threshold: float


TEST_CASES = [
    Case(
        masks_in=create_masks([(100, 100)], [50]),
        scores=torch.Tensor([1.0]),
        masks_out=create_masks([(100, 100)], [50]),
        iou_threshold=0.5,
    ),
    # IOU for this case: 0.59
    Case(
        masks_in=create_masks([(100, 100), (120, 100)], [50, 50]),
        scores=torch.Tensor([1.0, 0.5]),
        masks_out=create_masks([(100, 100)], [50]),
        iou_threshold=0.5,
    ),
    Case(
        masks_in=create_masks([(120, 100), (100, 100)], [50, 50]),
        scores=torch.Tensor([0.5, 1.0]),
        masks_out=create_masks([(100, 100)], [50]),
        iou_threshold=0.5,
    ),
    Case(
        masks_in=create_masks([(100, 100), (120, 100)], [50, 50]),
        scores=torch.Tensor([1.0, 0.5]),
        masks_out=create_masks([(100, 100), (120, 100)], [50, 50]),
        iou_threshold=0.7,
    ),
    Case(
        masks_in=create_masks([(120, 100), (100, 100)], [50, 50]),
        scores=torch.Tensor([0.5, 1.0]),
        masks_out=create_masks([(100, 100), (120, 100)], [50, 50]),
        iou_threshold=0.7,
    ),
]


@pt.mark.parametrize("iou_func", [pure_torch._compute_maskiou])
@pt.mark.parametrize("case", TEST_CASES)
def test_iou(iou_func, case):
    if len(case.masks_in) < 2:
        # skip single-mask test case
        return

    ious = iou_func(case.masks_in, case.masks_in).numpy()
    np.testing.assert_array_almost_equal(ious, [[1, 0.59], [0.59, 1]], decimal=2)


@pt.mark.parametrize("nms_func", [pure_torch.masknms])
@pt.mark.parametrize("case", TEST_CASES)
def test_masknms(nms_func, case):
    result = nms_func(
        case.masks_in, case.scores, iou_threshold=case.iou_threshold
    ).numpy()
    np.testing.assert_array_almost_equal(result, case.masks_out)
