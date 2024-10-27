import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.gta_loader import SegmentationDataset
from dataset.cityscapes_loader import CityscapesDataset_blockgen
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import PolynomialLR
import torch, gc
from ignite.handlers.param_scheduler import LRScheduler
import torch.nn.functional as F
import random
import piq
import numpy as np
import wandb
import os

# GPU 메모리 정리 및 설정
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_mse(model, val_loader):
    """
    주어진 모델에 대해 MSE를 사용하여 validation을 수행하는 함수.

    Args:
        model (torch.nn.Module): 평가할 모델
        val_loader (torch.utils.data.DataLoader): 검증 데이터셋의 DataLoader

    Returns:
        float: 전체 검증 데이터셋의 평균 MSE
    """
    # 모델을 평가 모드로 전환
    model.eval()
    mse_loss = nn.MSELoss()

    total_mse = 0
    total_samples = 0

    with torch.no_grad():  # 검증 시에는 그래디언트 계산을 하지 않음
        for iter, data in enumerate(val_loader):
            image, _ = data  # 배치 가져오기
            masked_image = randomTile_zero(image.to(device), (64, 64), 0.5)

            # Forward pass
            _, feature = model(masked_image, return_features=True)
            predicted_image = decoder(feature)

            # 배치의 MSE 계산
            batch_mse = mse_loss(predicted_image, image.to(device)).item()

            # MSE 누적
            total_mse += batch_mse * image.size(0)
            total_samples += image.size(0)

    # 평균 MSE 계산
    avg_mse = total_mse / total_samples

    return avg_mse



# 디코더 클래스 정의 (회전 검출을 위한 출력 계층)

class MaskReconstructionDecoder(nn.Module):
    def __init__(self):
        super(MaskReconstructionDecoder, self).__init__()

        # 업샘플링을 위한 ConvTranspose2d 레이어들
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

        # 최종 출력 레이어
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 업샘플링 단계와 활성화 함수
        x = F.relu(self.deconv1(x))  # [2, 2048, 65, 129] -> [2, 1024, 130, 258]
        x = F.relu(self.deconv2(x))  # [2, 1024, 130, 258] -> [2, 512, 260, 516]
        x = F.relu(self.deconv3(x))  # [2, 512, 260, 516] -> [2, 256, 520, 1032]
        x = F.relu(self.deconv4(x))  # [2, 256, 520, 1032] -> [2, 128, 1040, 2064]
        x = F.relu(self.deconv5(x))  # [2, 128, 1040, 2064] -> [2, 64, 2080, 4128]

        # 최종 해상도 [2, 3, 512, 1024]로 조정
        x = F.interpolate(x, size=(512, 1024), mode='bilinear', align_corners=True)
        x = self.final_conv(x)  # [2, 64, 512, 1024] -> [2, 3, 512, 1024]

        return x




def randomTile_zero(images, tile_size, zero_ratio=0.2):
    """
    배치의 모든 이미지에 정해진 크기의 타일로 나눈 후, 랜덤한 타일을 선택하여 비율에 맞게 픽셀 값을 0으로 설정하는 함수.

    Args:
        images (torch.Tensor): 이미지 텐서 (B, C, H, W) 형태, 보통 배치, 채널, 높이, 너비.
        tile_size (tuple): 타일의 크기 (tile_height, tile_width)
        zero_ratio (float): 타일 중 몇 개를 0으로 설정할지 결정하는 비율 (0과 1 사이)

    Returns:
        modified_images (torch.Tensor): 랜덤 타일이 0으로 설정된 이미지 텐서
    """
    # 배치 내 모든 이미지를 처리
    batch_size, channels, height, width = images.shape
    tile_height, tile_width = tile_size

    # 이미지 내 타일의 개수 계산
    num_tiles_y = height // tile_height
    num_tiles_x = width // tile_width

    if num_tiles_y == 0 or num_tiles_x == 0:
        raise ValueError("Tile size is too large for the given image dimensions.")

    # 전체 타일 중 선택된 비율만큼 0으로 만들 타일의 개수 계산
    total_tiles = num_tiles_y * num_tiles_x
    num_tiles_to_zero = int(total_tiles * zero_ratio)

    # 원본 이미지를 복사하여 수정
    modified_images = images.clone()

    for i in range(batch_size):
        # 랜덤하게 타일을 선택하기 위해 타일 인덱스를 섞음
        all_tile_indices = [(y, x) for y in range(num_tiles_y) for x in range(num_tiles_x)]
        np.random.shuffle(all_tile_indices)
        selected_tiles = all_tile_indices[:num_tiles_to_zero]

        # 선택된 타일의 픽셀 값을 0으로 설정
        for tile_y, tile_x in selected_tiles:
            top = tile_y * tile_height
            left = tile_x * tile_width
            modified_images[i, :, top:top + tile_height, left:left + tile_width] = 0

    return modified_images


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
model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
decoder = MaskReconstructionDecoder().to(device)

# 이미지에 적용할 변환 (ToTensor -> Normalize)
def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 1024)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet  정규화
    ])

# Cityscapes 데이터셋 준비 (훈련 및 검증)
train_dataset = CityscapesDataset_blockgen(
    images_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/leftImg8bit/train",
    transform=image_transform(),
    debug=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_config["batch_size"],
    shuffle=True
)

val_dataset = CityscapesDataset_blockgen(
    images_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/leftImg8bit/val",
    transform=image_transform(),
    debug=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=training_config["batch_size"],
    shuffle=False
)

# 옵티마이저 및 스케줄러 설정
optimizer = torch.optim.SGD(list(model.parameters()) + list(decoder.parameters()),
                            lr=training_config["initial_lr"],
                            weight_decay=training_config["optimizer"]["weight_decay"],
                            momentum=training_config["optimizer"]["momentum"])
pt_scheduler = PolynomialLR(optimizer, total_iters=training_config["max_iter"])


wandb.init(
    # set the wandb project where this run will be logged
    project="robotvision",
    )

# 훈련 루프
num_epochs = 3
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for iter, data in enumerate(train_loader):
        image, _ = data  # 배치 가져오기
        masked_image = randomTile_zero(image.to(device),(64,64),0.5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # 이미지 값 복원

        # Forward pass
        _, feature = model(masked_image, return_features=True)
        predicted_image = decoder(feature)
        predicted_image = predicted_image * std + mean
        image = image.to(device) * std + mean
        loss_total = 1 - piq.ssim(predicted_image, image, data_range=1.0)

        import pdb; pdb.set_trace()
        # Backpropagation 및 파라미터 업데이트
        optimizer.zero_grad()
        epoch_loss += loss_total.item()
        loss_total.backward()
        optimizer.step()

    pt_scheduler.step()
    # 모델 검증

    avg_mse = validate_mse(model, val_loader)
    torch.save(model.state_dict(), os.path.join(exp_dir, f"encoder-epoch_{epoch}.pth"))
    torch.save(decoder.state_dict(), os.path.join(exp_dir, f"decoder-epoch_{epoch}.pth"))
    wandb.log({"train_loss": epoch_loss, "valid/mse": avg_mse})

    # 성능 출력
    print(f"avgerage_MSE: {avg_mse:.4f}")

wandb.finish()
