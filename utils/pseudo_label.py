import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.cityscapes_loader import CityscapesDataset
import numpy as np
import os
import logging
from PIL import Image
from .colorize import colorize_mask
import time
import tqdm
device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_pseudo_labels(model, round_idx, save_round_eval_path, image_transform, args=None):
  logger = logging.getLogger('Cityscapes adaptation')

  """args"""
  target_image_dir = "/home/hyunho/sfda/data/cityscapes_dataset/leftImg8bit/train"
  target_mask_dir = "/home/hyunho/sfda/data/cityscapes_dataset/gtFine/train"
  pseudo_batch_size = 2
  debug = False
  kc_value = 'conf'
  num_classes = 19
  ds_rate = 10
  
  
  train_dataset = CityscapesDataset(
      images_dir=target_image_dir,
      masks_dir=target_mask_dir,
      image_transform=image_transform,
      debug=debug
  )

  train_loader = DataLoader(
      train_dataset, 
      batch_size=pseudo_batch_size, 
      shuffle=False, 
      pin_memory=True,
      num_workers=16
  )
  save_pred_vis_path = os.path.join(save_round_eval_path, 'pred_vis')
  save_prob_path = os.path.join(save_round_eval_path, 'prob')
  save_pred_path = os.path.join(save_round_eval_path, 'pred')
  if not os.path.exists(save_pred_vis_path):
      os.makedirs(save_pred_vis_path)
  if not os.path.exists(save_prob_path):
      os.makedirs(save_prob_path)
  if not os.path.exists(save_pred_path):
      os.makedirs(save_pred_path)

  pred_cls_num = np.zeros(num_classes)
  conf_dict = {k: [] for k in range(num_classes)}

  model.eval()
  start_eval = time.time()
  logger.info('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
  with torch.no_grad():
    for batch in tqdm.tqdm(train_loader):
      image, label, name = batch  # TODO 파일 저장하려면 name 나오도록 해야됨 (list of image name) : image, label, name = batch
      image = image.to(device)

      output = model(image).cpu().softmax(1)
      flipped_out = model(image.flip(-1)).cpu().softmax(1)

      output = 0.5 * (output + flipped_out.flip(-1))
      pred_prob, pred_labels = output.max(1)

      for b_ind in range(image.shape[0]):
        image_name = name[b_ind].split('/')[-1].split('.')[0]

        np.save('%s/%s.npy' % (save_prob_path, image_name), output[b_ind].numpy().transpose(1, 2, 0))
        if debug:
            colorize_mask(pred_labels[b_ind].numpy().astype(np.uint8)).save(
                '%s/%s_color.png' % (save_pred_vis_path, image_name))
        Image.fromarray(pred_labels[b_ind].numpy().astype(np.uint8)).save(
            '%s/%s.png' % (save_pred_path, image_name))
        

      if kc_value == 'conf':
        for idx_cls in range(num_classes):
          idx_temp = pred_labels == idx_cls
          pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + idx_temp.sum()
          if idx_temp.any():
            conf_cls_temp = pred_prob[idx_temp].numpy().astype(np.float32)[::ds_rate]
            conf_dict[idx_cls].extend(conf_cls_temp)

  model.train()
  logger.info(
    '###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time() - start_eval)
    )
  return conf_dict, pred_cls_num, save_prob_path, save_pred_path