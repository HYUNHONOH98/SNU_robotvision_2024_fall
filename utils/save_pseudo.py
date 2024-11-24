from PIL import Image
import os
import numpy as np

palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def save_pseudo_labels(names, pseudo_labels, save_dir):
  for idx, name in enumerate(names):
    img_name = name.split("/")[-1]
    if img_name in [
      "aachen_000000_000019_leftImg8bit.png",
      "bochum_000000_000313_leftImg8bit.png",
      "bremen_000000_000019_leftImg8bit.png",
      "cologne_000000_000019_leftImg8bit.png",
      "darmstadt_000000_000019_leftImg8bit.png",
      "dusseldorf_000000_000019_leftImg8bit.png",
      "erfurt_000000_000019_leftImg8bit.png",
      "hamburg_000000_000019_leftImg8bit.png",
      "hanover_000000_000019_leftImg8bit.png",
      "jena_000000_000019_leftImg8bit.png",
      "krefeld_000000_000019_leftImg8bit.png",
      "monchengladbach_000000_000019_leftImg8bit.png",
      "strasbourg_000000_000019_leftImg8bit.png",
      "tubingen_000000_000019_leftImg8bit.png",
      "ulm_000000_000019_leftImg8bit.png",
      "weimar_000000_000019_leftImg8bit.png",
      "zurich_000000_000019_leftImg8bit.png",
    ]:
      save_path = os.path.join(save_dir, img_name)

      tensor = pseudo_labels[idx].byte()
      array = tensor.cpu().numpy()
      height, width = array.shape
      rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

      for class_id, color in enumerate(palette):
        rgb_image[array == class_id] = color
      rgb_image[array == 255] = [255, 255, 255]
      
      image = Image.fromarray(rgb_image)
      image.save(save_path)
  return