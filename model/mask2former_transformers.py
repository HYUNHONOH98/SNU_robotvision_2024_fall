from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from .model_layer_match import map_mask2former_weight

def load_mask2former(device):
  cfg = Mask2FormerConfig()
  cfg.backbone_config.depths = [2, 2, 6, 2]
  cfg.num_labels = 19

  preprocessor = Mask2FormerImageProcessor(ignore_index=255, 
                                         do_reduce_labels=False, 
                                         do_resize=False,
                                         do_rescale=False, 
                                         do_normalize=False)
  
  model = Mask2FormerForUniversalSegmentation(cfg).to(device)
  model = map_mask2former_weight(model, device)

  return model, preprocessor

  