from .configs import MODEL as deeplab_config
from .backbone import ResNetV1c
from .decode_head import DepthwiseSeparableASPPHead, FCNHead
import torch.nn as nn
import pdb


class DeeplabV3PlusModel(nn.Module):
  def __init__(self):
      super().__init__()
      del deeplab_config["backbone"]["type"]
      del deeplab_config["decode_head"]["type"]
      del deeplab_config["auxiliary_head"]["type"]

      self.backbone = ResNetV1c(**deeplab_config["backbone"])
      self.decode_head = DepthwiseSeparableASPPHead(**deeplab_config["decode_head"])
      self.auxiliary_head = FCNHead(**deeplab_config["auxiliary_head"])
  
  def forward(self, x):
     x = self.backbone(x)
     x = self.decode_head(x)
     return x
    #  x = self.decode_head(x)
    #  return self.auxiliary_head(x)