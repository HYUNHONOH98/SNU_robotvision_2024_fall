import torch
from torch import nn
from torch.utils.data import DataLoader

from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, mIoU, ConfusionMatrix
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from dataset.gta_loader import SegmentationDataset
from torch.optim.lr_scheduler import PolynomialLR
from ignite.handlers.param_scheduler import LRScheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np

# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "exp_name": "exp/deeplabv2",
    "max_iter" : 40000,
    "batch_size" : 4,
    "initial_lr" : 2.5e-4,
    "optimizer" : {
        "name" : "SGD",
        "momentum" : 0.9,
        "weight_decay" : 5e-4
    },
    "lr_scheduler" : "polynomial"
}

from model.deeplabv2 import DeeplabMulti
model = DeeplabMulti(num_classes=19, pretrained=True).to(device)


image_transforms = transforms.Compose([
    # transforms.RandomResizedCrop(size =(1280, 720), ratio=(0.5, 2.0)),
    transforms.Resize((512,1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4422, 0.4379, 0.4246), std=(0.2572, 0.2516, 0.2467)),
    transforms.ColorJitter()

    # Add normalization if needed
])

# Mean: tensor([0.4422, 0.4379, 0.4246])
# Std: tensor([0.2572, 0.2516, 0.2467])

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
    num_workers=16
)


optimizer = torch.optim.SGD(model.parameters(), 
                            lr=training_config["initial_lr"],
                            weight_decay=training_config["optimizer"]["weight_decay"],
                            momentum=training_config["optimizer"]["momentum"])
pt_scheduler = PolynomialLR(optimizer,
                            total_iters=training_config["max_iter"])
scheduler = LRScheduler(pt_scheduler)
criterion = nn.CrossEntropyLoss(ignore_index=255)

trainer = create_supervised_trainer(model, optimizer, criterion, device)
trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

val_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion),
    "mIoU": mIoU(cm=ConfusionMatrix(num_classes=19))
}

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

log_interval = 10

@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f} Avg IoU: {metrics['mIoU']:.2f}")


def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    training_config["exp_name"],
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)

train_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=log_interval),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

tb_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)

trainer.run(train_loader, max_epochs=27)

tb_logger.close()
