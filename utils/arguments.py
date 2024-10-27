import argparse
import os.path as osp

BASE_DIR = "/home/hyunho/sfda"
save =  "exp_data_debug"
train_image_dir = "data/cityscapes_dataset/leftImg8bit/train"
valid_image_dir = "data/cityscapes_dataset/leftImg8bit/val"
valid_mask_dir = "data/cityscapes_dataset/gtFine/val"
pretrained_source_model_path = "exp/deeplabv2_1024/best_model_4_accuracy=0.8350.pt"
model_dir = "exp/pseudo_train_debug"
num_classes = 19
num_rounds = 3
init_tgt_portion = 0.2
max_tgt_portion = 0.5
tgt_port_step = 0.05
num_epoch = 2
train_batch_size = 1
initial_lr = 2.5e-4
weight_decay = 5e-4
momentum = 0.9
max_grad_norm = 1
alpha = 0.00
debug = False
poly_power = 0.9
cityscape_image_mean = (0.4422, 0.4379, 0.4246)
cityscape_image_std = (0.2572, 0.2516, 0.2467)
input_size = (720, 1280)
cal_entropy = "mean"
accumulation_steps = 2

# Pseudo labeling
target_image_dir =  "data/cityscapes_dataset/leftImg8bit/train"
target_mask_dir =  "data/cityscapes_dataset/gtFine/train"
pseudo_batch_size = 2
kc_value = 'conf'
ds_rate = 10

# loss
entropy_lambda = 0.005
rotation_lambda = 0.01
reconstruction_lambda = 0.01

def join_base_path(args):
    args.save = osp.join(BASE_DIR, args.save)
    args.train_image_dir = osp.join(BASE_DIR, args.train_image_dir)
    args.valid_image_dir = osp.join(BASE_DIR, args.valid_image_dir)
    args.valid_mask_dir = osp.join(BASE_DIR, args.valid_mask_dir)
    args.target_image_dir = osp.join(BASE_DIR, args.target_image_dir)
    args.target_mask_dir = osp.join(BASE_DIR, args.target_mask_dir)
    return args

def get_args():
    parser = argparse.ArgumentParser(description="SFDA experiment args", conflict_handler='resolve')
    parser.add_argument("--save", type=str, default=save,
                        help="pseudo label 데이터를 저장할 위치")
    parser.add_argument("--num_rounds", type=int, default= num_rounds, 
                        help= 3)
    parser.add_argument("--init_tgt_portion", type=float, default=init_tgt_portion, 
                        help= "The initial portion of target to determine kc")
    parser.add_argument("--max_tgt_portion", type=float, default=max_tgt_portion , 
                        help= 'The max portion of target to determine kc')
    parser.add_argument("--tgt_port_step", type=float, default= tgt_port_step, 
                        help= 'The portion step in target domain in every round of self-paced self-trained neural network')
    parser.add_argument("--num_epoch", type=int, default=num_epoch , 
                        help= "한 라운드 당 에폭 수")
    parser.add_argument("--train_image_dir", type=str, default= train_image_dir, 
                        help= "train image 를 가져올 폴더 경로" )
    parser.add_argument("--valid_image_dir", type=str, default= valid_image_dir, 
                        help= "valid image 를 가져올 폴더 경로")
    parser.add_argument("--train_batch_size", type=str, default= train_batch_size, 
                        help= 2)
    parser.add_argument("--valid_mask_dir", type=str, default= valid_mask_dir, 
                        help= "valid mask 를 가져올 폴더 경로")
    parser.add_argument("--initial_lr", type=float, default= initial_lr, 
                        help= "initial learning rate")
    parser.add_argument("--weight_decay", type=float, default= weight_decay, 
                        help= "SGD weight decay 정도" )
    parser.add_argument("--momentum", type=float, default= momentum, 
                        help= "SGD momentum 크기")
    parser.add_argument("--max_grad_norm", type=int, default= max_grad_norm, 
                        help= "grad norm 을 통한 다중 로스 위험 개선")
    parser.add_argument("--alpha", type=float, default= alpha, 
                        help= "teacher model EMA 진행할 때, 얼마나 teacher 모델에 반영할지 결정하는 수치.")
    parser.add_argument("--debug", action="store_true", 
                        help="디버깅 모드를 실행할지 결정. 옵션을 명시하면 True" )
    parser.add_argument("--model_dir", type=str, default= model_dir, 
                        help= "훈련된 모델을 저장할 폴더.")
    parser.add_argument("--pretrained_source_model_path", type=str, default= pretrained_source_model_path, 
                        help= "pretrained 된 source model (teacher, studetnt 를 initiating 하는데 사용됨)을 불러올 경로")
    parser.add_argument("--encoder_update_only", action="store_true",
                        help= "encoder parameter 만 훈련 시킨다.")
    parser.add_argument("--poly_power", type=float, default=poly_power,
                        help= "Poly LR scheduler 의 power 값")
    parser.add_argument("--target_image_dir", type=str, default=target_image_dir,
                        help= "target dataset image 를 가져올 경로")
    parser.add_argument("--target_mask_dir", type=str, default=target_mask_dir,
                        help= "target dataset mask 를 가져올 경로")
    parser.add_argument("--pseudo_batch_size", type=int, default=pseudo_batch_size,
                        help= "pseudo label 의 배치 사이즈")
    parser.add_argument("--kc_value", type=str, default=kc_value,
                        help= "kc 를 어떻게 진행할 건지?")
    parser.add_argument("--ds_rate", type=int, default=ds_rate,
                        help= "pseudo label downsampling rate")
    parser.add_argument("--num_classes", type=int, default=num_classes,
                        help= "target dataset class 개수")
    parser.add_argument("--cal_entropy", type=str, default=cal_entropy,
                        help= "Entropy Loss 계산 방식")
    
    parser.add_argument("--entropy_lambda", type=float, default=entropy_lambda,
                        help= "Entropy Loss 계수")
    parser.add_argument("--reconstruction_lambda", type=float, default=reconstruction_lambda,
                        help= "Entropy Loss 계수")
    parser.add_argument("--rotation_lambda", type=float, default=rotation_lambda,
                        help= "Entropy Loss 계수")
    parser.add_argument("--accumulation_steps", type=int, default=accumulation_steps,
                        help= "Gradient accumulation 정도")
    
    return join_base_path(parser.parse_known_args()[0])