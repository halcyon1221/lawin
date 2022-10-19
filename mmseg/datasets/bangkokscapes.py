import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class BangkokScapeDataset(CustomDataset):
  CLASSES = ('Road','Misc','Building','Tree','Car','Footpath','Motorcycle','Pole','Person','Trash','Crosswalk')
  PALETTE = [[128, 128, 0],[0, 0, 0],[128, 128, 128],[64, 0, 0],
            [128, 0, 0],[0, 0, 128],[0, 128, 0],[0, 128, 128],
            [64, 128, 0],[192, 0, 0],[128, 0, 128]]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None