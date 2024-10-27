import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


def suffix_file_search(path, suffix):
    dir = Path(path)
    return sorted([str(file) for file in dir.rglob(f"*{suffix}") if file.is_file()])

class CityscapesDataset(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, masks_dir, image_transform=None, both_transform=None, debug=False, image_suffix="_leftImg8bit.png", mask_suffix="_labelTrainIds.png", rotate_function=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.both_transform= both_transform
        self.debug = debug
        self.rotate_function = rotate_function
        self.cityscapes_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.labels =['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle']

        self.images = suffix_file_search(images_dir, image_suffix)
        self.masks = suffix_file_search(masks_dir, mask_suffix)
        if debug:
          self.images = self.images[:10]
          self.masks = self.masks[:10]

        assert len(self.images) == len(self.masks), "Number of images and masks should be equal."


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')


        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = np.array(Image.open(mask_path).convert('P'))

        # Convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.rotate_function:
            rotated_img, rotate_label = self.rotate_function(image)
            rotated_img = self.image_transform(rotated_img)
            rotated_img = self.both_transform(rotated_img)

        
        if self.image_transform:
            image = self.image_transform(image)
        if self.both_transform:
            image, mask = self.both_transform((image, mask))

        if self.rotate_function:
            return image, mask.squeeze(0), name, rotated_img, rotate_label
        else:
            return image, mask.squeeze(0), name
    


"""class CityscapesDataset2(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, transform=None, debug=False,
                 image_suffix="_leftImg8bit.png"):
        self.images_dir = images_dir
        self.transform = transform
        self.debug = debug
        self.cityscapes_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                   [190, 153, 153], [153, 153, 153], [250, 170,
                                                                      30], [220, 220, 0],
                                   [107, 142, 35], [152, 251, 152], [70, 130, 180],
                                   [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                                   [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle']

        self.images = suffix_file_search(images_dir, image_suffix)
        if debug:
            self.images = self.images[:10]


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        image, label = rotate_image(image)
        if self.transform:
            transformed_image = self.transform(image)

        return transformed_image, label


class CityscapesDataset3(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None, debug=False,
                 image_suffix="_leftImg8bit.png", mask_suffix="_labelTrainIds.png"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.debug = debug
        self.cityscapes_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                   [190, 153, 153], [153, 153, 153], [250, 170,
                                                                      30], [220, 220, 0],
                                   [107, 142, 35], [152, 251, 152], [70, 130, 180],
                                   [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                                   [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle']

        self.images = suffix_file_search(images_dir, image_suffix)
        self.masks = suffix_file_search(masks_dir, mask_suffix)
        if debug:
            self.images = self.images[:10]
            self.masks = self.masks[:10]
        assert len(self.images) == len(self.masks), "Number of images and masks should be equal."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = np.array(Image.open(mask_path).convert('P'))

        # Convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Rotate image
        rot_image, rot_label = rotate_image(image)


        if self.transform:
            transformed_image = self.transform(image)
            rot_transformed_image = self.transform(rot_image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return transformed_image, mask.squeeze(0), rot_transformed_image, rot_label"""


class CityscapesDataset_RV(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, masks_dir, transform=None, target_transform=None, crop_transform=None,
                 debug=False,
                 image_suffix="_leftImg8bit.png", mask_suffix="_labelTrainIds.png"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_transform = target_transform
        self.crop_transform = crop_transform
        self.debug = debug
        self.cityscapes_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                                   [190, 153, 153], [153, 153, 153], [250, 170,
                                                                      30], [220, 220, 0],
                                   [107, 142, 35], [152, 251, 152], [70, 130, 180],
                                   [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                                   [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
        self.labels = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                       'traffic light', 'traffic sign', 'vegetation', 'terrain',
                       'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                       'motorcycle', 'bicycle']

        self.images = suffix_file_search(images_dir, image_suffix)
        self.masks = suffix_file_search(masks_dir, mask_suffix)
        if debug:
            self.images = self.images[:10]
            self.masks = self.masks[:10]
        assert len(self.images) == len(self.masks), "Number of images and masks should be equal."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = np.array(Image.open(mask_path).convert('P'))

        # Convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Rotate image

        if self.crop_transform:
            image, mask = self.crop_transform(image, mask)

        rot_image, rot_label = rotate_image(image)

        if self.transform:
            transformed_image = self.transform(image)
            rot_transformed_image = self.transform(rot_image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return transformed_image, mask.squeeze(0), rot_transformed_image, rot_label


class CityscapesDataset_blockgen(Dataset):
    # def __init__(self, images_dir, masks_dir, transform: A.Compose =None, debug=False):
    def __init__(self, images_dir, transform=None, debug=False,
                 image_suffix="_leftImg8bit.png"):
        self.images_dir = images_dir
        self.transform = transform
        self.debug = debug
        self.images = suffix_file_search(images_dir, image_suffix)
        if debug:
            self.images = self.images[:10]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            transformed_image = self.transform(image)


        return transformed_image, name