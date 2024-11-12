import sys
sys.path.append("/home/hyunho/sfda/")
import torch
import torch, gc

gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_source_model_path = "exp/deeplabv2_1024/best_model_4_accuracy=0.8350.pt"
import os
pretrained_source_model_path= os.path.join("/home/hyunho/sfda", pretrained_source_model_path)
print(f"device : {device}")

from model.deeplabv2 import DeeplabMultiBayes

stu_model = DeeplabMultiBayes().to(device).train()
stu_model.resnet.load_state_dict(torch.load(pretrained_source_model_path, map_location=device, weights_only=True))

tut_model = DeeplabMultiBayes().to(device).eval()
tut_model.resnet.load_state_dict(torch.load(pretrained_source_model_path, map_location=device, weights_only=True))

import torch
from torch.utils.data import DataLoader
from dataset.cityscapes_loader import CityscapesDataset
from torchvision import transforms
from utils.data_augmentation import RandomResizedCropWithMask, RandomHorizontalFlip

cityscape_image_mean = (0.4422, 0.4379, 0.4246)
cityscape_image_std = (0.2572, 0.2516, 0.2467)
input_size = (720, 1280)

train_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)])
])
valid_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
])
train_both_transforms = transforms.Compose([
    RandomHorizontalFlip(0.5),
    RandomResizedCropWithMask(input_size)
    ])

train_dataset = CityscapesDataset(
    images_dir="/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/train",
    masks_dir="/home/hyunho/sfda/data/cityscapes_dataset/gtFine/train",
    image_transform=train_image_transforms,
    both_transform=train_both_transforms,
    debug=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=2, 
    shuffle=True, 
    pin_memory=True,
    num_workers=16
)
data = next(iter(train_loader))
img, label, name = data

img = img.to(device)
output, mu, var  = stu_model(img)

from utils.loss import kld_loss, calculate_pseudo_loss
kl_loss = kld_loss(mu, var)
import pdb; pdb.set_trace()

tut_outptut, _, _  = tut_model(img)

loss_ = calculate_pseudo_loss(tut_outptut, output, kl_loss["threshold"], 2)

import pdb; pdb.set_trace()