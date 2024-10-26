import torch.nn as nn
import torch.nn.functional as F


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

class EffReconstructionDecoder(nn.Module):
    def __init__(self, output_size = (720, 1280)):
        super(EffReconstructionDecoder, self).__init__()
        
        # [B, 2048, 65, 129] -> [B, 1024, 130, 258]
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # [B, 1024, 130, 258] -> [B, 512, 260, 516]
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # [B, 512, 260, 516] -> [B, 256, 512, 1024]
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # [B, 256, 512, 1024] -> [B, 3, 512, 1024] (Output RGB image)
        self.upconv4 = nn.Sequential(
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        return x
