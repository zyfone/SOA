# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset


from .watercolor import WatercolorDataset
from .voc_watercolor_test import WaterColorDataset_test

from .voc_openset import Openset_Voc
from .clipart_openset import Openset_Clipart
from .clipart_openset_test import Openset_Clipart_test
from .water2voc_test import Water2voc_test
__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset",
           "Openset_Voc","Openset_Clipart","Openset_Clipart_test",
           "WatercolorDataset","WaterColorDataset_test","Water2voc_test"]
