import os


def set_arg_path(save):
  save_path = save
  save_pseudo_label_path = os.path.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
  save_stats_path = os.path.join(save_path, 'stats')  # in 'save_path'
  save_lst_path = os.path.join(save_path, 'list')

  if not os.path.exists(save_path):
      os.makedirs(save_path)
  if not os.path.exists(save_pseudo_label_path):
      os.makedirs(save_pseudo_label_path)
  if not os.path.exists(save_stats_path):
      os.makedirs(save_stats_path)
  if not os.path.exists(save_lst_path):
      os.makedirs(save_lst_path)
  
  return save_pseudo_label_path, save_stats_path, save_lst_path

def set_arg_round_path(save, round_idx):
    save_round_eval_path = os.path.join(save, str(round_idx))
    save_pseudo_label_color_path = os.path.join(
        save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
    if not os.path.exists(save_round_eval_path):
        os.makedirs(save_round_eval_path)
    if not os.path.exists(save_pseudo_label_color_path):
        os.makedirs(save_pseudo_label_color_path)

    return save_round_eval_path, save_pseudo_label_color_path
