import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask, input_mask
########################################<---Tail--->

def color_jitter(color_jitter, data=None, s=0.2, p=0.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                data = seq(data)
    return data


def gaussian_blur(blur, data=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data


import torch
import random
import torchvision.transforms.functional as TF

def rotate_image(image):
    # 랜덤으로 회전 각도 선택
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    
    # 이미지 회전
    rotated_image = TF.rotate(image, angle)
    
    # 각도를 레이블로 사용
    label = angles.index(angle)
    label_tensor = torch.zeros(4)
    label_tensor[label] = 1


    return rotated_image, label_tensor
import torch
import random
import torchvision.transforms.functional as F

class RandomResizedCropWithMask:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        """Get parameters for cropping and resizing."""
        width, height = F.get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(random.uniform(*log_ratio))

            w = int(round(torch.sqrt(target_area * aspect_ratio)))
            h = int(round(torch.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = width / height
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img, mask):
        # Random crop parameters
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # Apply crop to both image and mask
        img = F.resized_crop(img, i, j, h, w, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask = F.resized_crop(mask, i, j, h, w, self.size, interpolation=F.InterpolationMode.NEAREST)

        return img, mask
