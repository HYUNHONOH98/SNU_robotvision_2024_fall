import torch
from torch import nn
from torch.utils.data import DataLoader
import evaluate
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, mIoU, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from dataset.gta_loader import SegmentationDataset
from torch.optim.lr_scheduler import PolynomialLR
from ignite.handlers.param_scheduler import LRScheduler
from torchvision import transforms
from model.model_layer_match import map_mask2former_weight
import torch, gc

gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "max_iter" : 40000,
    "batch_size" : 2,
    "initial_lr" : 0.0001,
    "optimizer" : {
        "name" : "AdamW", # AdamW
        # "momentum" : 0.9,
        "weight_decay" : 0.05,
        # lr multiplier 0.1
        # we decay the learning rate at 0.9 and 0.95 fractions 
        # of the total number of training steps by a factor of 10.
    },
    "lr_scheduler" : "polynomial",
    "train_name" : "Mask2Former"
}

from model.mask2former_transformers import load_mask2former
model, preprocessor = load_mask2former(device)


def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors="pt",
    )

    batch["original_segmap"] = segmentation_maps
    
    return batch


image_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0)),
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),

    # Add normalization if needed
])

# For the mask, we only need to resize and convert it to tensor
mask_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0),  interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Resize((512,1024), interpolation=transforms.InterpolationMode.NEAREST),
    # transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))),
    # transforms.ToTensor(),
])

train_dataset = SegmentationDataset(
    images_dir="/home/hyunho/sfda/data/gta5_dataset/images",
    masks_dir="/home/hyunho/sfda/data/gta5_dataset/labels",
    transform=image_transforms,
    target_transform=mask_transforms,
)


train_loader = DataLoader(
    train_dataset, 
    batch_size=training_config["batch_size"], 
    shuffle=True, 
    pin_memory=True,
    num_workers=16,
    collate_fn=collate_fn
)

from tqdm.auto import tqdm

def is_backbone_param(name):
    return 'pixel_level_module' in name

optimizer = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if is_backbone_param(n)], 'lr': 0.00001},
                               {'params': [p for n, p in model.named_parameters() if not is_backbone_param(n)], 'lr': 0.0001}],
                              weight_decay=training_config["optimizer"]["weight_decay"]
                              )

# Define total number of epochs and corresponding steps
total_epochs = 50
total_steps = len(train_loader) * total_epochs

# StepLR schedule where we decay at 90% and 95% of total training steps
decay_points = [int(0.9 * total_steps), int(0.95 * total_steps)]
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decay_points, gamma=0.1)

# mIoU
def eval_step(engine, batch):
    return batch

def answer_sheet(tensor):
    tensor[tensor == 19] = 255
    num_classes = 19  # 0~18까지의 값을 다루기 때문에 클래스는 19개
    mask = tensor != 255  # 255인 부분을 무시하기 위한 마스크
    one_hot = torch.zeros((num_classes, 512, 1024), dtype=torch.float32)
    tensor = tensor.clamp(0, num_classes - 1)  # 텐서에서 값이 19 이상이면 18로 클램핑

    # 유효한 값(255가 아닌 값)에 대해 one-hot 인코딩 수행
    one_hot.scatter_(0, tensor.unsqueeze(0), 1.0)

    # 255인 부분을 0으로 유지
    one_hot[:, ~mask] = 0

    return one_hot

# default_evaluator = Engine(eval_step)
# cm = ConfusionMatrix(num_classes=19)
# metric = mIoU(cm)
# metric.attach(default_evaluator, 'miou')

running_loss = 0.0
num_samples = 0
for epoch in range(50):
    preds = []
    gts = []
    print("Epoch:", epoch)
    model.train()
    for idx, batch in enumerate(train_loader):
        
        # Reset the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            mask_labels=[labels.to(device) for labels in batch["mask_labels"]],
            class_labels=[labels.to(device) for labels in batch["class_labels"]],
        )

        # Backward propagation
        loss = outputs.loss
        loss.backward()

        batch_size = batch["pixel_values"].size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx + 1 % 100 == 0:
            print(f"Iter : {idx + 1}\tLoss: {round(running_loss/num_samples, 4)}")

        # Optimization
        optimizer.step()

        # if idx > 1000: continue
        # # calculate mIoU
        # model.eval()
        
        # # Forward pass
        # with torch.no_grad():
        #     # get original images
        #     target_sizes = [(image.shape[-2], image.shape[-1]) for image in batch["pixel_values"]]
        #     # predict segmentation maps
        #     predicted_segmentation_maps = preprocessor.post_process_semantic_segmentation(outputs,
        #                                                                                 target_sizes=target_sizes)

        #     # get ground truth segmentation maps
        #     preds.extend([answer_sheet(x) for x in predicted_segmentation_maps])
        #     gts.extend(list(batch["original_segmap"]))
        # # import pdb; pdb.set_trace()
        
        # state = default_evaluator.run([[pred, gt] for pred, gt in list(zip(preds, gts))])
        # print("Mean IoU:", state.metrics['miou'])
