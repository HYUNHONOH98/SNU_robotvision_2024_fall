import torch
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from dataset.gta_loader import SegmentationDataset

import PIL.Image as Image
import numpy as np

import torch, gc

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



image_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # Add normalization if needed
])

mask_transforms = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64)))
])

train_dataset = SegmentationDataset(
    images_dir="/home/hyunho/sfda/data/gta5_dataset/images/train",
    masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels/train",
    transform=image_transforms,
    target_transform=mask_transforms
)

train_loader = DataLoader(
    train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers = 16
)

def compute_mean_std(dataloader):
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    num_batches = 0
    num_pixels = 0

    for images, _ in dataloader:
        import pdb; pdb.set_trace()
        # images shape: [batch_size, 3, height, width]
        num_batches += 1
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_squared_sum / num_pixels) - (mean ** 2))
    return mean, std

mean, std = compute_mean_std(train_loader)
print('Mean:', mean)
print('Std:', std)