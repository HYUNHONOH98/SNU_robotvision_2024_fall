import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.gta_loader import SegmentationDataset
from dataset.cityscapes_loader import CityscapesDataset2
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import PolynomialLR
import torch, gc
from ignite.handlers.param_scheduler import LRScheduler
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (2048, 1, 1)
        self.flatten = nn.Flatten()  # (2048)
        self.fc = nn.Linear(2048, 4)  # (4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

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
decoder = Decoder()

image_transforms = transforms.Compose([
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),
])
mask_transforms = transforms.Compose([
    transforms.Resize((512,1024), interpolation=transforms.InterpolationMode.NEAREST)
])

# GTA5 dataset
train_dataset = CityscapesDataset2(
  images_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/leftImg8bit/train",
  transform = image_transforms,
  debug=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=training_config["batch_size"], 
    shuffle=True
)

# TODO
optimizer = torch.optim.SGD(model.parameters(),
                            lr=training_config["initial_lr"],
                            weight_decay=training_config["optimizer"]["weight_decay"],
                            momentum=training_config["optimizer"]["momentum"])
pt_scheduler = PolynomialLR(optimizer,
                            total_iters=training_config["max_iter"])
scheduler = LRScheduler(pt_scheduler)
criterion = nn.CrossEntropyLoss()



model.train()

for iter, data in enumerate(train_loader):
  image, label = data
  output, features = model(image, return_features=True)
  output2 = decoder(features)
  result = criterion(output2, label)

  import pdb; pdb.set_trace()


# 어디에 회전 함수를 넣어야할지
# encoder feature 는 어떻게 받아와야할지
# decoder 는 어떻게 만들어야 할지