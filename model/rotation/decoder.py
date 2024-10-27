import torch.nn as nn


# 디코더 클래스 정의 (회전 검출을 위한 출력 계층)
class RotationClassifierDecoder(nn.Module):
    def __init__(self):
        super(RotationClassifierDecoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (2048, 1, 1)
        self.flatten = nn.Flatten()  # (2048)
        self.fc = nn.Linear(2048, 4)  # (4)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
