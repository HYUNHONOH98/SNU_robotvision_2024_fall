import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.gta_loader import SegmentationDataset
from dataset.cityscapes_loader import CityscapesDataset
from torch.optim.lr_scheduler import PolynomialLR
from ignite.handlers.param_scheduler import LRScheduler
from torchvision import transforms
import os
from utils.pseudo_label import generate_pseudo_labels
from utils.set_path import set_arg_path, set_arg_round_path
from utils.data_augmentation import RandomResizedCropWithMask, RandomHorizontalFlip, rotate_image
import utils.label_selection as label_selection
from utils.model_parameter_ema import update_model_params
from model.deeplabv2 import DeeplabMulti
from utils.arguments import get_args
from tqdm import tqdm
from utils.loss import iou, HLoss
from model.reconstruction.masking import randomTile_zero
from model.reconstruction.decoder import EffReconstructionDecoder
from model.rotation.decoder import RotationClassifierDecoder
import piq
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

def train_one_epoch(model, optimizer, data_loader, args=None, recon_decoder=None, rotate_decoder=None):
    data_mean = torch.tensor(cityscape_image_mean).view(1, 3, 1, 1).to(device)
    data_std = torch.tensor(cityscape_image_std).view(1, 3, 1, 1).to(device)

    total_loss = 0.0

    # Loss 선언
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    rotation_loss = nn.CrossEntropyLoss()
    entropy_loss = HLoss(mode=args.cal_entropy)
    l1_loss = torch.nn.SmoothL1Loss()

    model.train()
    tot_iter = len(data_loader)

    model.zero_grad()       
    optimizer.zero_grad()                            
    for i, data in tqdm(enumerate(data_loader)):
        img, label, _, rotated_img, rotate_label = data
        img = img.to(device)
        label = label.to(device)
        rotated_img = rotated_img.to(device)
        rotate_label = rotate_label.to(device)

        output, feature = model(img, return_features=True)

        # basic loss
        loss = ce_loss(output, label.long())

        if args.entropy_lambda > 0.:
            loss += entropy_loss(output) * args.entropy_lambda

        if args.reconstruction_lambda > 0.:
            masked_img = randomTile_zero(img,(64,64),0.5)
            _, masked_feature = model(masked_img, return_features=True)
            masked_output = recon_decoder(masked_feature) # AUX decoder for reconstruction

            unnormed_img = (img * data_std + data_mean).clamp(0, 1)
            unnormed_output = (masked_output * data_std + data_mean).clamp(0, 1)

            ssim_loss = 1 - piq.ssim(unnormed_output, unnormed_img, data_range=1.0)
            # ssim_loss = l1_loss(unnormed_output, unnormed_img)

            loss += ssim_loss * args.reconstruction_lambda
        
        if args.rotation_lambda > 0.:
            _, feature = model(rotated_img, return_features=True)
            rotated_output = rotate_decoder(feature)
            rotated_output = rotated_output
            rotate_label = rotate_label.argmax(dim=1).long()

            rot_ce_loss = rotation_loss(rotated_output, rotate_label)

            loss += rot_ce_loss * args.rotation_lambda

        loss /= args.accumulation_steps     
        loss.backward()

        if (i+1) % args.accumulation_steps == 0:             # Wait for several backward steps
            optimizer.step()                            # Now we can do an optimizer step
            optimizer.zero_grad()          

        total_loss += loss.item()
        if (i+1) % 100 == 0:
            print('iter = {} of {} completed, loss = {:.4f}'.format(i+1, tot_iter, loss.item()))

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


    # 실험에 필요한 폴더들 생성
    save_pseudo_label_path, save_stats_path, save_lst_path = set_arg_path(args.save)

    ## model 초기화.
    teacher_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
    teacher_model.load_state_dict(torch.load(args.pretrained_source_model_path, map_location=device, weights_only=True))

    student_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
    student_model.load_state_dict(torch.load(args.pretrained_source_model_path, map_location=device, weights_only=True))
    
    ## AUX decoders 초기화.
    rotation_decoder = RotationClassifierDecoder().to(device)
    rotation_decoder.train()
    reconstruction_decoder = EffReconstructionDecoder(input_size).to(device)
    reconstruction_decoder.train()

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
    pseudo_image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cityscape_image_mean, std=cityscape_image_std),
        transforms.Resize(input_size)
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

    tgt_portion = args.init_tgt_portion

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
                            # total_iters=5,
                            power=args.poly_power )
    student_model.freeze_encoder_bn()

    for round_idx in range(args.num_rounds):
        # round 마다 pseudo label 저장
        save_round_eval_path, save_pseudo_label_color_path = set_arg_round_path(args.save, round_idx)

        if round_idx == 0 or args.alpha > 0 or args.debug == True: # EMA 가 진행되지 않는다면 pseudo label 을 계속 재생성 할 필요 없음 (teacher model freeze)
        # pseudo label 생성 및 저장. 
            conf_dict, pred_cls_num, save_prob_path, save_pred_path  = generate_pseudo_labels(teacher_model, round_idx, save_round_eval_path, pseudo_image_transform, args)

        cls_thresh = label_selection.kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, args)
        label_selection.label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args)
        
        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_portion)

        train_dataset = CityscapesDataset(
            images_dir=args.train_image_dir,
            masks_dir=save_pseudo_label_path,
            image_transform=train_image_transforms,
            both_transform=train_both_transforms,
            mask_suffix="_leftImg8bit.png",
            debug=args.debug,
            rotate_function=rotate_image
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
            train_loss = train_one_epoch(student_model, optimizer, train_dataloader, args, reconstruction_decoder, rotation_decoder)
            valid_score = validate_one_epoch(student_model, valid_dataloader)
            scheduler.step()

            if not args.debug:
                wandb.log({"train_loss": train_loss, "valid_score": valid_score})
            
        if args.alpha > 0:
            # EMA 로 teacher model update 하기.
            update_model_params(teacher_model, student_model, args.alpha)

        print("모델 업데이트 끝.")
        if valid_score > new_record:
            torch.save(student_model.state_dict(), os.path.join(args.model_dir,f'student-round_{round_idx}-IOU-{valid_score}.pth'))
            torch.save(teacher_model.state_dict(), os.path.join(args.model_dir,f'teacher-round_{round_idx}-IOU-{valid_score}.pth'))
            new_record = valid_score
    
    if not args.debug:
        wandb.finish()
        
        
      



if __name__=="__main__":
        
        main()
