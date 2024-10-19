from dataset.gta_loader import SegmentationDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import torch, gc
from model.losses.cross_entropy_loss import cross_entropy
from torchvision.transforms import InterpolationMode
gc.collect()
torch.cuda.empty_cache()


image_transforms = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    # Add normalization if needed
])

# For the mask, we only need to resize and convert it to tensor
mask_transforms = transforms.Compose([
    transforms.Resize((512, 1024), interpolation= InterpolationMode.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64)))
])

train_dataset = SegmentationDataset(
    images_dir="/home/hyunho/sfda/data/gta5_dataset/images/train",
    masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels/train",
    transform=image_transforms,
    target_transform=mask_transforms,
    debug=True
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=1, 
    shuffle=True,
    pin_memory = True
)

from model.modeling import deeplabv3_resnet50

model = deeplabv3_resnet50(num_classes=19,
                   output_stride=4,
                   pretrained_backbone=True)
model.eval()

for data in train_loader:
  image, label = data
  import pdb; pdb.set_trace()
  pred = model(image)
  import pdb; pdb.set_trace()
  loss = cross_entropy(pred, label,ignore_index=255)
  import pdb; pdb.set_trace()
