import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.gta_loader import SegmentationDataset
from torchvision import transforms
import numpy as np

import torch, gc
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "max_iter" : 40000,
    "batch_size" : 4,
    "initial_lr" : 2.5e-4,
    "optimizer" : {
        "name" : "SGD",
        "momentum" : 0.9,
        "weight_decay" : 5e-4
    },
    "lr_scheduler" : "polynomial"
}

from model.deeplabv2 import DeeplabMulti
model = DeeplabMulti(num_classes=19, pretrained=True)


image_transforms = transforms.Compose([
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),
])
mask_transforms = transforms.Compose([
    transforms.Resize((512,1024), interpolation=transforms.InterpolationMode.NEAREST)
])

# GTA5 dataset
train_dataset = SegmentationDataset(
    images_dir="/home/hyunho/sfda/data/gta5_dataset/images",
    masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels",
    transform=image_transforms,
    target_transform=mask_transforms,
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=training_config["batch_size"], 
    shuffle=True
)

# TODO
# optimizer 정의
# loss function 정의

model.train()

for iter, data in enumerate(train_loader):
  image, label = data
  output = model(image)


# 어디에 회전 함수를 넣어야할지
# encoder feature 는 어떻게 받아와야할지
# decoder 는 어떻게 만들어야 할지