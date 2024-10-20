import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

image_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0)),
    # transforms.Resize((1280, 720)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),
    transforms.RandomHorizontalFlip(p=0.5),

    # Add normalization if needed
])
# For the mask, we only need to resize and convert it to tensor
mask_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0),  interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.Resize((1280, 720), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64)))
])





class SegmentationDataset(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None, debug=False, model="resnet"):
        self.model = model
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform= target_transform
        self.debug = debug
        self.gta_pallete = [(128, 64, 128),(244, 35, 232),(70, 70, 70),(102, 102, 156),(190, 153, 153),(153, 153, 153),(250, 170, 30),(220, 220, 0),(107, 142, 35),(152, 251, 152),(70, 130, 180),(220, 20, 60),(255, 0, 0),(0, 0, 142),(0, 0, 70),(0, 60, 100),(0, 80, 100),(0, 0, 230),(119, 11, 32)]
        # self.gta_pallete = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],[107, 142, 35], [152, 251, 152], [70, 130, 180],[220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],[0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.labels =["road","sidewalk","building","wall","fence","pole","light","sign","vegetation","terrain","sky","person","rider","car","truck","bus","train","motocycle","bicycle"]
        # Get list of image and mask file names
        if debug:
            self.images = sorted(os.listdir(images_dir)[:10])
            self.masks = sorted(os.listdir(masks_dir)[:10])
        else:
            self.images = sorted(os.listdir(images_dir))
            self.masks = sorted(os.listdir(masks_dir))

        assert len(self.images) == len(self.masks), "Number of images and masks should be equal."
        self.palette_index_to_class = {0: 255,1: 255,2: 255,3: 255,4: 255,5: 255,6: 255,7: 0,8: 1,9: 255,10: 255,11: 2,12: 3,13: 4,14: 255,15: 255,16: 255,17: 5,18: 5,19: 6,20: 7,21: 8,22: 9,23: 10,24: 11,25: 12,26: 13,27: 14,28: 15,29: 255,30: 255,31: 16,32: 17,33: 18,34: 13}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')


        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('P')

        # Map mask indices to class indices
        mask_class = map_mask_indices(mask, self.palette_index_to_class)

        # Convert to tensor
        mask_class = torch.from_numpy(mask_class).unsqueeze(0)
        
        if self.transform:
            transformed_image = self.transform(image)
        if self.target_transform:
            mask_class = self.target_transform(mask_class)

        return transformed_image, mask_class.squeeze(0)
    
        # def create_palette_mapping(self):
    #     # Load a sample mask to get the palette
    #     palette_index_to_class = {}
    #     collected_idx = []

    #     for i in range(self.__len__()):
    #         sample_mask_path = os.path.join(self.masks_dir, self.masks[i])
    #         sample_mask = Image.open(sample_mask_path).convert('P')
    #         palette = sample_mask.getpalette()
    #         palette_rgb = [tuple(palette[i:i+3]) for i in range(0, len(palette), 3)]

    #         # Create the mapping
    #         for idx, color in enumerate(palette_rgb):
    #             if color in self.gta_pallete:
    #                 class_idx = self.gta_pallete.index(color)
    #                 palette_index_to_class[idx] = class_idx

    #                 if class_idx not in collected_idx:
    #                     collected_idx.append(class_idx)

    #             else:
    #                 palette_index_to_class[idx] = 255  # Ignore index

    #         if len(collected_idx) == 19:
    #             break
    #         elif i > 100:
    #             break
    #     self.i = i
    #     self.collected_idx = collected_idx

    #     return palette_index_to_class

def map_mask_indices(mask_array, index_mapping):
    # Vectorized mapping of mask indices to class indicess
    mapped_mask = np.vectorize(index_mapping.get)(mask_array)
    return mapped_mask.astype(np.int64)


