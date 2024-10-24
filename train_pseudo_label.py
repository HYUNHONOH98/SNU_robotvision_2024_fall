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
from utils.loggers import set_logger
import utils.label_selection as label_selection
from utils.model_parameter_ema import update_model_params
from model.deeplabv2 import DeeplabMulti
from tqdm import tqdm
from utils.loss import iou, HLoss
import numpy as np
import torch, gc
import wandb


gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = set_logger("/home/hyunho/sfda/exp_data", 'training logger', False)


def train_one_epoch(model, optimizer, data_loader, args=None):
    """args"""
    entropy_lambda = 0.005
    total_loss = 0.0
    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    entropy_loss = HLoss()

    model.train()
    tot_iter = len(data_loader)
    for i, data in tqdm(enumerate(data_loader)):
        optimizer.zero_grad()
        img, label, _ = data
        output = model(img.to(device))

        l1 = ce_loss(output, label.long().to(device))
        l2 = entropy_loss(output)

        loss = l1 + l2 * entropy_lambda
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i+1, tot_iter, loss.item()))
    
    return total_loss

def validate_one_epoch(model, dataloader, args=None):
    model.eval()
    total_score = np.zeros(19)
    tot_iter = len(dataloader)
    for i, data in tqdm(enumerate(dataloader)):
        img, label, _ = data
        output = model(img.to(device))

        _, classes = output.softmax(1).max(1)
        classes = classes.detach().cpu()
        score = iou(classes, label.long(), C=19, ignore=255)
        total_score += score

    total_score = total_score / tot_iter
    mIoU = round(np.mean(total_score),2)

    logger.info(f'validation completed. mIoU : {mIoU}')
    return mIoU


def main():

    wandb.init(
    # set the wandb project where this run will be logged
    project="robotvision",
    )



    """args"""
    save = "/home/hyunho/sfda/exp_data"
    num_rounds = 3
    init_tgt_portion = 0.2
    max_tgt_portion = 0.5
    tgt_port_step = 0.05
    num_epoch = 2
    train_image_dir = "/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/train"
    valid_image_dir = "/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/val"
    valid_mask_dir = "/home/hyunho/sfda/data/cityscapes_dataset/gtFine/val"
    train_batch_size = 2
    initial_lr = 2.5e-4
    weight_decay = 5e-4
    momentum = 0.9
    max_grad_norm = 1
    alpha = 0.01
    debug = False
    model_dir = "/home/hyunho/sfda/exp/pseudo_train_2"

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


    # 실험에 필요한 폴더들 생성
    save_pseudo_label_path, save_stats_path, save_lst_path = set_arg_path(save)

    ## model 초기화.
    teacher_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
    teacher_model.load_state_dict(torch.load("/home/hyunho/sfda/exp/deeplabv2_1022/best_model_3_accuracy=0.8210.pt", map_location=device, weights_only=True))

    student_model = DeeplabMulti(num_classes=19, pretrained=True).to(device)
    student_model.load_state_dict(torch.load("/home/hyunho/sfda/exp/deeplabv2_1022/best_model_3_accuracy=0.8210.pt", map_location=device, weights_only=True))

    # transforms 정의
    image_transforms = transforms.Compose([
    transforms.Resize((720,1280)),
    transforms.ToTensor(),
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize((720,1280), interpolation=transforms.InterpolationMode.NEAREST)
    ])


    valid_dataset = CityscapesDataset(
        images_dir=valid_image_dir,
        masks_dir=valid_mask_dir,
        transform=image_transforms,
        target_transform=mask_transforms,
        debug=debug
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )

    tgt_portion = init_tgt_portion
    for round_idx in range(num_rounds):
        # round 마다 pseudo label 저장
        save_round_eval_path, save_pseudo_label_color_path = set_arg_round_path(save, round_idx)

        # pseudo label 생성 및 저장
        conf_dict, pred_cls_num, save_prob_path, save_pred_path  = generate_pseudo_labels(teacher_model, round_idx, save_round_eval_path, image_transforms, mask_transforms)
        cls_thresh = label_selection.kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx)
        label_selection.label_selection(cls_thresh,round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path)

        tgt_portion = min(tgt_portion + tgt_port_step, max_tgt_portion)

        train_dataset = CityscapesDataset(
            images_dir=train_image_dir,
            masks_dir=save_pseudo_label_path,
            transform=image_transforms,
            target_transform=mask_transforms,
            mask_suffix="_leftImg8bit.png",
            debug=debug
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=16
        )

        # Encoder training ONLY
        # params = [ 
        #         {'params': student_model.layer1.parameters(), 'lr': initial_lr * 1.0},
        #         {'params': student_model.layer2.parameters(), 'lr': initial_lr * 1.0},
        #         {'params': student_model.layer3.parameters(), 'lr': initial_lr * 1.0},
        #         {'params': student_model.layer4.parameters(), 'lr': initial_lr * 1.0},
        #         ]
        # To clip grad norm
        # param_only = []
        # for param in params:
        #      param_only.extend(param["params"])

        params = student_model.parameters()

        optimizer = torch.optim.SGD(
            params, 
            lr=initial_lr,
            weight_decay=weight_decay,
            momentum=momentum
        )
        
        nn.utils.clip_grad_norm_(
            params,
            max_grad_norm
        )
        scheduler = PolynomialLR(optimizer,
                                total_iters=len(train_dataloader))
        student_model.freeze_bn()
        

        
        for epoch in range(num_epoch):
            print(f"Epoch {epoch}")
            train_loss = train_one_epoch(student_model, optimizer, train_dataloader)
            valid_score = validate_one_epoch(student_model, valid_dataloader)
            scheduler.step()

            wandb.log({"train_loss": train_loss, "valid_score": valid_score})
        
        update_model_params(teacher_model, student_model, alpha)
        print("모델 업데이트 끝.")

        
        torch.save(student_model.state_dict(), os.path.join(model_dir,f'student_{round_idx}.pth'))
        torch.save(teacher_model.state_dict(), os.path.join(model_dir,f'teacher_{round_idx}.pth'))
    
    wandb.finish()
        
        
      



if __name__=="__main__":
        main()
