import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, mIoU, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from dataset.gta_loader import SegmentationDataset
from torch.optim.lr_scheduler import PolynomialLR
from ignite.handlers.param_scheduler import LRScheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np
import yaml

import torch, gc
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
cfg = Mask2FormerConfig()
cfg.backbone_config.depths = [2, 2, 6, 2]
cfg.num_labels = 19
# cfg.use_pretrained_backbone = True

preprocessor = Mask2FormerImageProcessor(ignore_index=255, 
                                         do_reduce_labels=False, 
                                         do_resize=False,
                                         do_rescale=False, 
                                         do_normalize=False)
def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )
    
    return batch

model = Mask2FormerForUniversalSegmentation(cfg).to(device)
model.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")


image_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0)),
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),

    # Add normalization if needed
])

# Mean: tensor([0.4422, 0.4379, 0.4246])
# Std: tensor([0.2572, 0.2516, 0.2467])

# For the mask, we only need to resize and convert it to tensor
mask_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0),  interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Resize((512,1024), interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
    # transforms.ToTensor(),
])

train_dataset = SegmentationDataset(
    images_dir="/home/hyunho/sfda/data/gta5_dataset/images",
    masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels",
    transform=image_transforms,
    target_transform=mask_transforms,
)


train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    pin_memory=True,
    num_workers=16,
    collate_fn=collate_fn
)

batch = next(iter(train_loader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k,v.shape)
  else:
    print(k,v[0].shape)


for batch in train_loader:
  # image, label = data
  output = model(
          pixel_values=batch["pixel_values"],
          mask_labels=[labels for labels in batch["mask_labels"]],
          class_labels=[labels for labels in batch["class_labels"]],
    )
  
  import pdb; pdb.set_trace()
