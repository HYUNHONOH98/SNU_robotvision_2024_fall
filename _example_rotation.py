import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.gta_loader import SegmentationDataset
from dataset.cityscapes_loader import CityscapesDataset_RV
from torchvision import transforms
import numpy as np
from torch.optim.lr_scheduler import PolynomialLR
import torch, gc
from ignite.handlers.param_scheduler import LRScheduler
import torch.nn.functional as F
import random

# 랜덤 시드를 사용하여 이미지와 마스크를 동일하게 자르는 클래스
class RandomResizedCropWithSeed:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        # 마스크 변환에 NEAREST 보간을 적용하여, 픽셀 값이 왜곡되지 않도록 설정
        self.image_transform = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio)
        self.mask_transform = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=transforms.InterpolationMode.NEAREST)

    def __call__(self, img, mask):
        # 동일한 시드를 사용하여 이미지와 마스크를 변환
        seed = random.randint(0, 10000)

        # 같은 시드로 랜덤 크롭 적용
        random.seed(seed)
        img_transformed = self.image_transform(img)

        random.seed(seed)
        mask_transformed = self.mask_transform(mask)

        return img_transformed, mask_transformed

# IoU (Intersection over Union) 계산 함수
def calculate_iou(pred_mask, true_mask, num_classes):
    # 마스크를 1차원으로 변환하여 간단하게 계산
    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    iou_scores = []
    for cls in range(num_classes):
        # 배경(255)은 무시
        if cls == 255:
            continue

        # 각 클래스별로 교집합과 합집합 계산
        intersection = ((pred_mask == cls) & (true_mask == cls)).sum().item()
        union = ((pred_mask == cls) | (true_mask == cls)).sum().item()

        if union == 0:
            iou_scores.append(float('nan'))  # 합집합이 없을 경우 NaN으로 처리
        else:
            iou_scores.append(intersection / union)

    # NaN을 무시하고 평균 IoU 반환
    return torch.tensor(iou_scores).nanmean().item()

# 모델 검증 함수 (평균 IoU와 회전 정확도 계산)
def validate_model(model, val_loader, num_classes):
    model.eval()  # 모델을 평가 모드로 설정
    total_iou = 0
    count = 0
    correct_rotations = 0
    total_rotations = 0
    with torch.no_grad():  # 검증 시에는 그래디언트 계산 필요 없음
        for data in val_loader:
            image, mask, rot_image, rot_label = data  # 검증 데이터셋에서 이미지와 마스크 가져오기
            output = model(image)  # 모델의 예측 출력
            pred_mask = output.argmax(dim=1)  # 예측 마스크 (argmax로 클래스 인덱스 추출)

            # IoU 계산
            iou = calculate_iou(pred_mask, mask, num_classes)
            total_iou += iou
            count += 1

            # Rotation detection 검증
            _, rot_feature = model(rot_image, return_features=True)
            rot_probability = decoder(rot_feature)
            pred_rotation = rot_probability.argmax(dim=1)
            rot_label_index = rot_label.argmax(dim=1)
            correct_rotations += (pred_rotation == rot_label_index).sum().item()
            total_rotations += rot_label.size(0)

    # 평균 IoU와 회전 검출 정확도 반환
    avg_iou = total_iou / count
    rotation_accuracy = correct_rotations / total_rotations

    return avg_iou, rotation_accuracy

# GPU 메모리 정리 및 설정
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디코더 클래스 정의 (회전 검출을 위한 출력 계층)
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

# 모델 초기화
from model.deeplabv2 import DeeplabMulti
model = DeeplabMulti(num_classes=19, pretrained=True)
decoder = Decoder()

# 이미지에 적용할 변환 (ToTensor -> Normalize)
def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

# 마스크에 적용할 변환 (Resize -> ToTensor)
def mask_transform():
    return transforms.Compose([
        transforms.Resize((512, 1024), interpolation=transforms.InterpolationMode.NEAREST),
    ])

# Cityscapes 데이터셋 준비 (훈련 및 검증)
train_dataset = CityscapesDataset_RV(
    images_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/leftImg8bit/train",
    masks_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/gtFine/train",
    transform=image_transform(),
    target_transform=mask_transform(),
    crop_transform=RandomResizedCropWithSeed((512, 1024)),
    debug=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=training_config["batch_size"],
    shuffle=True
)

val_dataset = CityscapesDataset_RV(
    images_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/leftImg8bit/val",
    masks_dir="C:/Users/박종선/PycharmProjects/pythonProject/SNU_robotvision_2024_fall/data/cityscapes_dataset/gtFine/val",
    transform=image_transform(),
    target_transform=mask_transform(),
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
scheduler = LRScheduler(pt_scheduler)

# 훈련 루프
num_epochs = 3
for epoch in range(num_epochs):
    for iter, data in enumerate(train_loader):
        image, mask, rot_image, rot_label = data  # 배치 가져오기

        # Forward pass
        output1 = model(image)
        predicted_mask = output1.softmax(1).argmax(1)

        output2, feature = model(rot_image, return_features=True)
        result_rotate = decoder(feature)
        rot_label_index = rot_label.argmax(dim=1).long()

        # 손실 계산 (세그멘테이션 + 회전 검출)
        loss1 = F.cross_entropy(output1, mask.long(), ignore_index=255)
        loss2 = F.cross_entropy(result_rotate, rot_label_index)
        loss_total = loss1 + loss2

        # Backpropagation 및 파라미터 업데이트
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        import pdb; pdb.set_trace()  # 디버깅을 위해 추가

    # 모델 검증
    iou_score, Rotation_accuracy = validate_model(model, val_loader, num_classes=19)

    # 성능 출력
    print(f"Mean IoU: {iou_score:.4f}")
    print(f"Rotation_accuracy: {Rotation_accuracy*100:.4f}%")
