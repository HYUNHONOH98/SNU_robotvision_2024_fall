import torch
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from torch import nn
from torch.utils.data import DataLoader
"from dataset.gta_loader import SegmentationDataset"
from dataset.cityscapes_loader import CityscapesDataset
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import PolynomialLR
import torch, gc
from ignite.handlers.param_scheduler import LRScheduler
import torch.nn.functional as F
import wandb
import tqdm
import random
import os
from utils.data_augmentation import RandomResizedCropWithMask, RandomHorizontalFlip, rotate_image
from scipy import linalg
from torchvision.models import inception_v3
from torchvision import transforms
from ignite.metrics import FID
from ignite.engine import Engine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fid_metric = FID()
# GPU 메모리 정리 및 설정
gc.collect()
torch.cuda.empty_cache()

def evaluation_step(engine, batch):
    with torch.no_grad():
        # batch에서 필요한 요소만 언패킹
        real_images = batch[0]  # 첫 번째 요소가 이미지라고 가정
        z = torch.randn(real_images.shape[0], latent_dim)
        fake_images = generator(z)
        return fake_images, real_images

evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")

#Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, batch_size):
        super(Generator, self).__init__()

        self.batch_size = batch_size
        self.init_size = (45, 80)  # 초기 특징 맵 크기

        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size[0] * self.init_size[1]))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Upsample(size=(720, 1280)),  # 최종 크기로 조정
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(self.batch_size, 128, self.init_size[0], self.init_size[1])
        img = self.conv_blocks(out)
        return img
#Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # 공통 컨볼루션 레이어
        self.common = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False)
        )

        # Segmentation 브랜치
        self.seg_branch = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(720, 1280), mode='bilinear', align_corners=False)
        )

        # Real/Fake 판별 브랜치
        self.rf_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # 공통 특징 추출
        features = self.common(x)

        # Segmentation map 생성
        seg_map = self.seg_branch(features)

        # Real/Fake 판별
        rf_pred = self.rf_branch(features)

        return seg_map, rf_pred

# 훈련 설정 파라미터
training_config = {
    "max_iter": 40000,
    "batch_size": 2,    # 종선 컴퓨터의 메모리에 맞게 조정
    "initial_lr": 2.5e-4,
    "optimizer": {
        "name": "SGD",
        "momentum": 0.9,
        "weight_decay": 5e-4
    },
    "lr_scheduler": "polynomial"
}

home_dir = "C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall"#"/home/hyunho/sfda"
exp_dir = "C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/exp/MaskReconstruction"#"/home/hyunho/sfda/exp/rotation"
debug = False

# 모델 초기화
from model.deeplabv2 import DeeplabMulti
latent_dim=100
generator = Generator(latent_dim, training_config["batch_size"])

model = DeeplabMulti(num_classes=19, pretrained=True)

discriminator = Discriminator(num_classes=19)

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
pseudo_image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
    transforms.Resize(input_size)
])
train_both_transforms = transforms.Compose([
    RandomHorizontalFlip(0.5),
    RandomResizedCropWithMask(input_size)
])

# Cityscapes 데이터셋 준비 (훈련 및 검증)
train_dataset = CityscapesDataset(
    images_dir=f"{home_dir}/data/cityscapes_dataset/leftImg8bit/train",
    masks_dir=f"{home_dir}/data/cityscapes_dataset/gtFine/train",
    image_transform=train_image_transforms,
    both_transform=train_both_transforms,
    debug=True,
    rotate_function=None
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=training_config["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=0
)

val_dataset = CityscapesDataset(
    images_dir=f"{home_dir}/data/cityscapes_dataset/leftImg8bit/train",
    masks_dir=f"{home_dir}/data/cityscapes_dataset/gtFine/train",
    image_transform=train_image_transforms,
    both_transform=train_both_transforms,
    debug=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_config["batch_size"],
    shuffle=False
)
ce_loss = nn.CrossEntropyLoss(ignore_index=255)
g_optimizer = torch.optim.SGD(generator.parameters(), lr=training_config["initial_lr"],
                            weight_decay=training_config["optimizer"]["weight_decay"],
                            momentum=training_config["optimizer"]["momentum"])

# 옵티마이저 및 스케줄러 설정
ed_optimizer = torch.optim.SGD(list(model.parameters()) + list(discriminator.parameters()),
                            lr=training_config["initial_lr"],
                            weight_decay=training_config["optimizer"]["weight_decay"],
                            momentum=training_config["optimizer"]["momentum"])
scheduler_ED = PolynomialLR(ed_optimizer, total_iters=training_config["max_iter"])
scheduler_G = PolynomialLR(g_optimizer, total_iters=training_config["max_iter"])
lambda_seg = 1
"""
wandb.init(
    # set the wandb project where this run will be logged
    project="robotvision",
    )
"""

# 훈련 루프
num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for iter, data in tqdm.tqdm(enumerate(train_dataloader)):
        image_Real, mask, _ = data
        z = torch.randn(training_config["batch_size"], latent_dim)
        image_Fake = generator(z)

        # Forward pass
        _, feature_Real = model(image_Real, return_features=True)
        _, feature_Fake = model(image_Fake, return_features=True)
        output_Real, pred_Real = discriminator(feature_Real)
        output_Fake, pred_Fake = discriminator(feature_Fake)

        real_label = torch.full((training_config["batch_size"], 1), 1, dtype=torch.float32)
        fake_label = torch.full((training_config["batch_size"], 1), 0, dtype=torch.float32)

        # 손실 계산
        D_real_loss = F.binary_cross_entropy_with_logits(pred_Real, real_label)
        D_fake_loss = F.binary_cross_entropy_with_logits(pred_Fake, fake_label)
        D_seg_loss = F.cross_entropy(output_Real, mask.long(), ignore_index=255)
        D_loss = D_real_loss + D_fake_loss + lambda_seg * D_seg_loss

        # Discriminator 업데이트
        ed_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        ed_optimizer.step()

        # Generator 손실 계산 및 업데이트
        _, feature_Fake = model(image_Fake, return_features=True)
        output_Fake, pred_Fake = discriminator(feature_Fake)
        G_fake_loss = F.binary_cross_entropy_with_logits(pred_Fake, real_label)
        G_loss = G_fake_loss

        g_optimizer.zero_grad()
        G_loss.backward()
        g_optimizer.step()

        epoch_loss += (D_loss.item() + G_loss.item())

    epoch_loss /= len(train_dataloader)
    scheduler_G.step()
    scheduler_ED.step()

    # 모델 검증
    evaluator.run(val_loader)
    fid_score = evaluator.state.metrics['fid']
    print(f"Epoch {epoch + 1}, FID: {fid_score:.4f}")

"""   torch.save(model.state_dict(), os.path.join(exp_dir, f"encoder-epoch_{epoch}.pth"))
    torch.save(decoder.state_dict(), os.path.join(exp_dir, f"decoder-epoch_{epoch}.pth"))

    # 성능 출력

wandb.finish()
"""