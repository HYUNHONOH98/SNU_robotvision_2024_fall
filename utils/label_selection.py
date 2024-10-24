import logging
import os
import shutil
import time

import numpy as np
from PIL import Image
from .colorize import colorize_mask



### FROM https://github.com/idiap/model-uncertainty-for-adaptation
logger = logging.getLogger('Cityscapes adaptation')

def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, args=None):
    """args"""
    num_classes = 19


    print('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    cls_thresh = np.ones(num_classes, dtype=np.float32)
    cls_sel_size = np.zeros(num_classes, dtype=np.float32)
    cls_size = np.zeros(num_classes, dtype=np.float32)

    for idx_cls in np.arange(0, num_classes):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True)  # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(len_cls * tgt_portion)
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            conf_dict[idx_cls] = None
    
    print("Per class thresholds:")
    print(cls_thresh)
    print('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(
        round_idx, time.time() - start_kc))
    return cls_thresh



def label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args= None):
    """args"""
    debug = True


    print('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    filenames = [os.path.splitext(x)[0] for x in os.listdir(save_prob_path) if x.endswith('npy')]
    for sample_name in filenames:
        probmap_path = os.path.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_path = os.path.join(save_pred_path, '{}.png'.format(sample_name))

        pred_prob = np.load(probmap_path)
        pred_label_labelIDs = np.asarray(Image.open(pred_path))

        save_wpred_vis_path = os.path.join(save_round_eval_path, 'weighted_pred_vis')
        if not os.path.exists(save_wpred_vis_path):
            os.makedirs(save_wpred_vis_path)
        weighted_prob = pred_prob / cls_thresh
        weighted_prob_ids = weighted_prob.argmax(axis=2).astype(np.uint8)

        if debug:
            colorize_mask(weighted_prob_ids).save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))

        weighted_conf = weighted_prob.max(axis=2)
        pred_label_labelIDs = weighted_prob_ids
        pred_label_labelIDs[weighted_conf < 1] = 255  # '255' in cityscapes indicates 'unlabaled' for trainIDs

        if debug:
            pseudo_label_col = colorize_mask(pred_label_labelIDs)
            pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
        
        pseudo_label_save = Image.fromarray(pred_label_labelIDs.astype(np.uint8))
        pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))

    
    shutil.rmtree(save_prob_path)
    print('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time() - start_pl))