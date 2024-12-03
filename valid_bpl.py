import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.cityscapes_loader import CityscapesDataset
from torch.optim.lr_scheduler import PolynomialLR
from torchvision import transforms
import os
from utils.data_augmentation import RandomResizedCropWithMask, RandomHorizontalFlip, rotate_image
from utils.model_parameter_ema import update_model_params
from utils.save_pseudo import save_valid_label
from model.deeplabv2 import DeeplabMulti, DeeplabMultiBayes
from model.gan import discriminator_loss, generator_loss, Generator
from utils.arguments import get_args
from tqdm import tqdm
from utils.loss import iou, HLoss, kld_loss, calculate_pseudo_loss
import numpy as np
import torch, gc
import wandb


gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()

"""args"""
cityscape_image_mean = (0.4422, 0.4379, 0.4246)
cityscape_image_std = (0.2572, 0.2516, 0.2467)
input_size = (720, 1280)


def validate_one_epoch(model, dataloader, args, epoch):
    model.eval()
    total_score = np.zeros(19)
    tot_iter = len(dataloader)
    for i, data in tqdm(enumerate(dataloader)):
        img, label, name = data
        output = model(img.to(device))["output"]

        _, classes = output.softmax(1).max(1)
        classes = classes.detach().cpu()

        save_dir = os.path.join("/home/hyunho/sfda/valid_label", args.exp_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, str(epoch))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_valid_label(name, classes, save_dir)

        score = iou(classes, label.long(), C=19, ignore=255)
        total_score += score

    total_score = total_score / tot_iter
    mIoU = round(np.mean(total_score),2)

    print(f'validation completed. mIoU : {mIoU}')
    return mIoU


def main():
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    student_model = DeeplabMultiBayes(num_classes=19, pretrained=True).to(device)
    student_model.load_state_dict(torch.load(args.pretrained_source_model_path, map_location=device, weights_only=True))

    valid_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
    ])


    valid_dataset = CityscapesDataset(
        images_dir=args.valid_image_dir,
        masks_dir=args.valid_mask_dir,
        image_transform=valid_image_transforms,
        debug=args.debug
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )

    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch}")
        valid_score = validate_one_epoch(student_model, valid_dataloader, args, epoch)


if __name__=="__main__":
        
        main()
