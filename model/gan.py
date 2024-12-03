import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def discriminator_loss(pred_Real, pred_Fake, args):
  real_label = torch.full((args.train_batch_size, 1), 1, dtype=torch.float32).to(device)
  fake_label = torch.full((args.train_batch_size, 1), 0, dtype=torch.float32).to(device)

  D_real_loss = F.binary_cross_entropy_with_logits(pred_Real, real_label)
  D_fake_loss = F.binary_cross_entropy_with_logits(pred_Fake, fake_label)

  return D_real_loss + D_fake_loss

def generator_loss(pred_Fake, args):
  real_label = torch.full((args.train_batch_size, 1), 1, dtype=torch.float32).to(device)

  G_fake_loss = F.binary_cross_entropy_with_logits(pred_Fake, real_label)

  return G_fake_loss


import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.init_size = (45, 80)  # 초기 특징 맵 크기
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size[0] * self.init_size[1]))

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
        batch_size = z.size(0)  # 입력 z의 첫 번째 차원으로 batch_size 계산
        out = self.l1(z)
        out = out.view(batch_size, 128, self.init_size[0], self.init_size[1])
        img = self.conv_blocks(out)
        return img