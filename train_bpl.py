import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.cityscapes_loader import CityscapesDataset
from torch.optim.lr_scheduler import PolynomialLR
from torchvision import transforms
import os
from utils.data_augmentation import RandomResizedCropWithMask, RandomHorizontalFlip, rotate_image
from utils.model_parameter_ema import update_model_params
from model.deeplabv2 import DeeplabMulti, DeeplabMultiBayes
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

def train_one_epoch(stu_model, tut_model, optimizer, data_loader, args=None, epoch=0):
    total_loss = 0.0

    # Loss 선언
    # entropy_loss = HLoss(mode=args.cal_entropy)

    stu_model.train()
    tut_model.eval()
    tot_iter = len(data_loader)

    stu_model.zero_grad()       
    optimizer.zero_grad()                            
    for i, data in tqdm(enumerate(data_loader)):
        img, _, name = data
        img = img.to(device)

        stu_output, mu, logvar = stu_model(img, return_features=True)
        tut_output = tut_model(img)

        # calculate KL loss
        kl_loss = kld_loss(mu1=mu, logvar1=logvar)

        pseudo_loss = calculate_pseudo_loss(tut_output, 
                                            stu_output, 
                                            kl_loss["threshold"], 
                                            {"name" : name, "epoch" : epoch}, 
                                            args)

        loss = 0.1 * kl_loss["loss"] + pseudo_loss["loss"]

        loss /= args.accumulation_steps     
        loss.backward()

        if (i+1) % args.accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()          

        total_loss += loss.item()
        if (i+1) % 100 == 0:
            print('iter = {} of {} completed, kl loss = {:.4f}, pseudo loss = {:.4f}'.format(i+1, tot_iter, kl_loss['loss'].item(), pseudo_loss['loss'].item()))

    avg_loss = round(total_loss / tot_iter, 2)
    return avg_loss

def validate_one_epoch(model, dataloader, args=None):
    model.eval()
    total_score = np.zeros(19)
    tot_iter = len(dataloader)
    for i, data in tqdm(enumerate(dataloader)):
        img, label, _ = data
        output = model(img.to(device))

        _, classes = output.softmax(1).max(1)
        classes = classes.detach().cpu()
        # TODO : 중간 아웃풋 10개씩 뽑기. (visualize 10 images 함수로 따로 만들어도 괜찮을 듯)
        score = iou(classes, label.long(), C=19, ignore=255)
        total_score += score

    total_score = total_score / tot_iter
    mIoU = round(np.mean(total_score),2)

    print(f'validation completed. mIoU : {mIoU}')
    return mIoU


def main():
    if not args.debug:
        wandb.init(
        # set the wandb project where this run will be logged
        project="robotvision",
        config=vars(args)
        )


    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)


    ## model 초기화.
    teacher_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
    teacher_model.load_state_dict(torch.load(args.pretrained_source_model_path, map_location=device, weights_only=True))

    student_model = DeeplabMultiBayes(num_classes=19, pretrained=True).to(device)
    student_model.resnet.load_state_dict(torch.load(args.pretrained_source_model_path, map_location=device, weights_only=True))
    
    # transforms 정의
    train_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)]),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)])
    ])
    valid_image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
    ])
    train_both_transforms = transforms.Compose([
        RandomHorizontalFlip(0.5),
        RandomResizedCropWithMask(input_size)
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

    nn.utils.clip_grad_norm_(
        student_model.parameters(),
        args.max_grad_norm
    )
    optimizer = torch.optim.SGD(
            student_model.parameters(), 
            lr=args.initial_lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum
        )
    scheduler = PolynomialLR(optimizer,
                            power=args.poly_power )
    student_model.resnet.freeze_encoder_bn()

        
    train_dataset = CityscapesDataset(
        images_dir=args.train_image_dir,
        masks_dir=args.train_mask_dir,
        image_transform=train_image_transforms,
        both_transform=train_both_transforms,
        debug=args.debug,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16
    )

    new_record = 0.0
    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch}")
        train_loss = train_one_epoch(student_model, teacher_model, optimizer, train_dataloader, args, epoch)
        valid_score = validate_one_epoch(student_model, valid_dataloader)
        scheduler.step()

        if not args.debug:
            wandb.log({"train_loss": train_loss, "valid_score": valid_score})
        
        if args.alpha > 0:
            # EMA 로 teacher model update 하기.
            update_model_params(teacher_model, student_model, args.alpha)

        print("모델 업데이트 끝.")
        if valid_score > new_record:
            torch.save(student_model.state_dict(), os.path.join(args.model_dir,f'student-{epoch}-IOU-{valid_score}.pth'))
            torch.save(teacher_model.state_dict(), os.path.join(args.model_dir,f'teacher-{epoch}-IOU-{valid_score}.pth'))
            new_record = valid_score

if not args.debug:
    wandb.finish()
        
        
      



if __name__=="__main__":
        
        main()
