import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, mIoU, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from dataset.gta_loader import SegmentationDataset
from dataset.cityscapes_loader import CityscapesDataset
from torchvision import transforms
import cv2
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

from model.modeling import deeplabv3_resnet50
model = deeplabv3_resnet50(num_classes=19,
                   output_stride=4,
                   pretrained_backbone=True).to(device)

from model.deeplabv2 import DeeplabMulti
model = DeeplabMulti(num_classes=19, pretrained=False)
model.load_state_dict(
  torch.load("/home/hyunho/sfda/exp/deeplabv2_1022/best_model_3_accuracy=0.8210.pt", map_location=device, weights_only=True)
)


image_transforms = transforms.Compose([
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),
])
mask_transforms = transforms.Compose([
    transforms.Resize((512,1024), interpolation=transforms.InterpolationMode.NEAREST)
])

# GTA5 dataset
# train_dataset = SegmentationDataset(
#     images_dir="/home/hyunho/sfda/data/gta5_dataset/images",
#     masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels",
#     transform=image_transforms,
#     target_transform=mask_transforms,
# )
# Cityscapes dataset
train_dataset = CityscapesDataset(
    images_dir="/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/train",
    masks_dir="/home/hyunho/sfda/data/cityscapes_dataset/gtFine/train",
    transform=image_transforms,
    target_transform=mask_transforms,
    debug=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=training_config["batch_size"], 
    shuffle=True, 
    pin_memory=True,
    num_workers=16
)

# state_dict = torch.load("/home/hyunho/sfda/_resnet/best_model_6_accuracy=0.8114.pt", map_location=device, weights_only=True)
# model.load_state_dict(state_dict)
model.eval()
import matplotlib.pyplot as plt

with torch.no_grad():
  for iter, data in enumerate(train_loader):
    img, label = data
    output = model(img)
    predicted_labels = torch.argmax(output, dim=1)
    
    # Select the first example from the batch
    predicted_image = predicted_labels[0].cpu().numpy()
    label_image = label[0].cpu().numpy()
    input_img = img[0].permute(1, 2, 0).cpu().numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

    # Plotting both the predicted and ground truth labels side by side
    plt.figure(figsize=(10, 5))

    # Predicted image
    plt.subplot(1, 3, 1)
    plt.imshow(predicted_image, cmap='nipy_spectral')  # You can choose another colormap
    plt.title('Predicted Segmentation')
    plt.axis('off')

    # Ground truth label
    plt.subplot(1, 3, 2)
    plt.imshow(label_image,  cmap='nipy_spectral')
    plt.title('Ground Truth Label')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(input_img)
    plt.title('Input Image')
    plt.axis('off')

    plt.savefig(f"/home/hyunho/sfda/_example/image/{iter}_city_deeplabv2.png")
    

    import pdb; pdb.set_trace()