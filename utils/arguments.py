import argparse

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
cityscape_image_mean = (0.4422, 0.4379, 0.4246)
cityscape_image_std = (0.2572, 0.2516, 0.2467)
input_size = (720, 1280)
pretrained_source_model_path = "/home/hyunho/sfda/exp/deeplabv2_1022/best_model_3_accuracy=0.8210.pt"


def common_args():
    parser = argparse.ArgumentParser(description="SFDA experiment args", conflict_handler='resolve')
    parser.add_argument("--save", type=str, default=save,
                        help="실험 데이터를 저장할 위치")
    parser.add_argument("--num_rounds", type=int, default= num_rounds, 
                        help= )
    parser.add_argument("--init_tgt_portion", type=float, default=init_tgt_portion, 
                        help= )
    parser.add_argument("--max_tgt_portion", type=int, default=max_tgt_portion , 
                        help= )
    parser.add_argument("--tgt_port_step", type=int, default= tgt_port_step, 
                        help= )
    parser.add_argument("--num_epoch", type=int, default=num_epoch , 
                        help= )
    parser.add_argument("--train_image_dir", type=int, default= train_image_dir, 
                        help= )
    parser.add_argument("--valid_image_dir", type=int, default= valid_image_dir, 
                        help= )
    parser.add_argument("--train_batch_size", type=int, default= train_batch_size, 
                        help= )
    parser.add_argument("--valid_mask_dir", type=int, default= valid_mask_dir, 
                        help= )
    parser.add_argument("--initial_lr", type=int, default= initial_lr, 
                        help= )
    parser.add_argument("--weight_decay", type=int, default= weight_decay, 
                        help= )
    parser.add_argument("--momentum", type=int, default= momentum, 
                        help= )
    parser.add_argument("--max_grad_norm", type=int, default= max_grad_norm, 
                        help= )
    parser.add_argument("--alpha", type=int, default= alpha, 
                        help= )
    parser.add_argument("--debug", type=int, default= debug, 
                        help= )
    parser.add_argument("--model_dir", type=int, default= model_dir, 
                        help= )
    parser.add_argument("--pretrained_source_model_path", type=int, default= debug, 
                        help= )
    parser.add_argument("--debug", type=int, default= debug, 
                        help= )
    parser.add_argument("--debug", type=int, default= debug, 
                        help= )
    

    return parser.parse_known_args()[0]