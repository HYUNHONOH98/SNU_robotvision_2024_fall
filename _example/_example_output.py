import sys
sys.path.append("/home/hyunho/sfda/")

import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.cityscapes_loader import CityscapesDataset
from torchvision import transforms
import os
from model.deeplabv2 import DeeplabMulti
from utils.colorize import colorize_mask
from tqdm import tqdm
from utils.loss import iou, HLoss
import numpy as np
import torch, gc

gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_one_epoch(model, dataloader, args=None):
    model.eval()
    total_score = np.zeros(19)
    tot_iter = len(dataloader)

    visualize_imgs = []
    scores = []


    for i, data in tqdm(enumerate(dataloader)):
        img, label, _ = data
        output = model(img.to(device))

        _, classes = output.softmax(1).max(1)
        classes = classes.detach().cpu()
        # TODO : 중간 아웃풋 10개씩 뽑기. (visualize 10 images 함수로 따로 만들어도 괜찮을 듯)
        score = iou(classes, label.long(), C=19, ignore=255)
        total_score += score

        visualize_imgs.append((classes, label))
        scores.append(score)

    total_score = total_score / tot_iter
    mIoU = round(np.mean(total_score),2)

    print(f'validation completed. mIoU : {mIoU}')
    return mIoU, visualize_imgs


"""args"""
cityscape_image_mean = (0.4422, 0.4379, 0.4246)
cityscape_image_std = (0.2572, 0.2516, 0.2467)
input_size = (720, 1280)



"""
VER  7
"""
# model_path = "/home/hyunho/sfda/exp/pseudo_train_7/student-round_0-IOU-30.47.pth"
# model_path = "/home/hyunho/sfda/exp/pseudo_train_7/student-round_1-IOU-30.37.pth"
model_path = "/home/hyunho/sfda/exp/pseudo_train_7/student-round_2-IOU-29.47.pth"


"""
VER 8 : rotation, reconstruction
"""
# model_path = "/home/hyunho/sfda/exp/pseudo_train_8/student-round_0-IOU-34.69.pth"
# model_path = "/home/hyunho/sfda/exp/pseudo_train_8/student-round_1-IOU-34.52.pth"
# model_path = "/home/hyunho/sfda/exp/pseudo_train_8/student-round_2-IOU-31.03.pth"

"""
VER 9 : rotation
"""
# model_path = "/home/hyunho/sfda/exp/pseudo_train_9/student-round_0-IOU-25.92.pth"
# model_path = "/home/hyunho/sfda/exp/pseudo_train_9/student-round_1-IOU-33.75.pth"
# model_path = "/home/hyunho/sfda/exp/pseudo_train_9/student-round_2-IOU-26.17.pth"

student_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
student_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

# transforms 정의
train_image_transforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std)
])



valid_dataset = CityscapesDataset(
    images_dir="/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/val",
    masks_dir="/home/hyunho/sfda/data/cityscapes_dataset/gtFine/val",
    image_transform=train_image_transforms,
    # both_transform=train_both_transforms,
    debug=True,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    num_workers=16
)

mIoU, visualize_imgs = validate_one_epoch(student_model, valid_dataloader)
image_path = "/home/hyunho/sfda/_example/image/example_output"

import matplotlib.pyplot as plt

exp_name = model_path.split("/")[-2]
if not os.path.exists(os.path.join(image_path, exp_name)):
   os.mkdir(os.path.join(image_path, exp_name))

round_num = model_path.split("/")[-1].split(".")[0]
if not os.path.exists(os.path.join(image_path, exp_name, round_num)):
   os.mkdir(os.path.join(image_path, exp_name, round_num))
for id, img in enumerate(visualize_imgs):
  pred, label = img

  color_pred = colorize_mask(pred[0].numpy().astype(np.uint8))
  color_label = colorize_mask(label[0].numpy().astype(np.uint8))

  plt.figure(figsize=(10, 5))
  
  plt.subplot(1,2,1)
  plt.imshow(color_pred)
  plt.title("prediction from student")
  plt.axis("off")

  plt.subplot(1,2,2)
  plt.imshow(color_label)
  plt.title("label")
  plt.axis("off")

  plt.savefig('%s/%s/%s/%s_pred.png' % (image_path, exp_name, round_num, id))